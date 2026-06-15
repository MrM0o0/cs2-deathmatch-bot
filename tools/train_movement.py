"""Train the behavioural-cloning movement model and export it to ONNX.

Runs on Google Colab (free GPU, dodges the AMD card) or locally on CPU (slow).
Input is a recording dir made by record_movement.py + prep_movement_labels.py:
each frame (90x160 BGR jpg) paired with a label row [w, a, s, d, crouch, turn].

The model is a small CNN with two heads:
  - keys:  5 independent sigmoids (w, a, s, d, crouch)  -> BCEWithLogits
  - turn:  5-way softmax (hardL, left, straight, right, hardR) -> CrossEntropy
Class imbalance (mostly idle / mostly straight) is countered with pos_weight on
the keys and class weights on the turn head.

Colab usage:
    # zip the session locally:  data/movement/<ts>/  (frames/ + actions.jsonl + labels.npy)
    # upload the zip, then:
    !pip -q install onnx
    !python train_movement.py --data /content/<ts> --epochs 15 --out movement.onnx

Local usage:
    pip install torch onnx
    python tools/train_movement.py --data data/movement/<ts> --epochs 15
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

H, W = 90, 160
TURN_CLASSES = 5


class MoveNet(nn.Module):
    """Small CNN: frame -> (key logits [w,a,s,d,crouch], turn logits [5])."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6)),
            nn.Flatten(),
        )
        self.trunk = nn.Sequential(nn.Linear(64 * 4 * 6, 128), nn.ReLU(), nn.Dropout(0.3))
        self.keys = nn.Linear(128, 5)
        self.turn = nn.Linear(128, TURN_CLASSES)

    def forward(self, z):
        z = self.trunk(self.features(z))
        return self.keys(z), self.turn(z)


def export_onnx(net, out_path):
    """Export to ONNX. Uses the dynamo exporter (handles adaptive pooling)."""
    net.eval().cpu()
    dummy = torch.zeros(1, 3, H, W)
    torch.onnx.export(
        net,
        dummy,
        out_path,
        input_names=["frame"],
        output_names=["keys_logits", "turn_logits"],
        dynamic_axes={"frame": {0: "batch"}},
        opset_version=18,
        dynamo=True,
    )
    print(f"[train] exported {out_path}")


def load_dataset(data_dir):
    labels = np.load(os.path.join(data_dir, "labels.npy"))  # (N, 6)
    frames_dir = os.path.join(data_dir, "frames")
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(frames_dir, "*.jpg")))
    assert len(names) == len(labels), f"{len(names)} frames vs {len(labels)} labels"
    x = np.empty((len(names), H, W, 3), dtype=np.uint8)
    for i, name in enumerate(names):
        x[i] = cv2.imread(os.path.join(frames_dir, name))
    return x, labels


def main():
    ap = argparse.ArgumentParser(description="Train the BC movement model")
    ap.add_argument("--data", required=True, help="Recording dir (frames/ + labels.npy)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="movement.onnx")
    args = ap.parse_args()

    from torch.utils.data import DataLoader, TensorDataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device: {dev}")

    x, y = load_dataset(args.data)
    print(f"[train] {len(x)} samples")
    # NHWC uint8 -> NCHW float [0,1]
    xt = torch.from_numpy(x).float().div_(255).permute(0, 3, 1, 2).contiguous()
    keys_t = torch.from_numpy(y[:, :5]).float()
    turn_t = torch.from_numpy(y[:, 5]).long()

    n = len(xt)
    perm = torch.randperm(n)
    n_val = max(1, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = DataLoader(
        TensorDataset(xt[tr_idx], keys_t[tr_idx], turn_t[tr_idx]),
        batch_size=args.batch,
        shuffle=True,
    )
    va = DataLoader(
        TensorDataset(xt[val_idx], keys_t[val_idx], turn_t[val_idx]), batch_size=args.batch
    )

    # Imbalance weights from the training split -- GENTLE on purpose. Full
    # inverse-frequency weighting made the model collapse to always predicting
    # the most-upweighted rare class (constant hard-left). Cap key weights and
    # use sqrt-inverse-frequency (also capped) for the turn classes.
    key_rate = keys_t[tr_idx].mean(0).clamp(0.05, 0.95)
    pos_weight = ((1 - key_rate) / key_rate).clamp(max=3.0).to(dev)
    turn_counts = torch.bincount(turn_t[tr_idx], minlength=TURN_CLASSES).float()
    inv_freq = turn_counts.sum() / (TURN_CLASSES * turn_counts.clamp(min=1))
    turn_weight = inv_freq.sqrt().clamp(max=3.0).to(dev)
    print(f"[train] key pos_weight={pos_weight.tolist()}")
    print(f"[train] turn weight={turn_weight.tolist()}")

    net = MoveNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    key_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    turn_loss = nn.CrossEntropyLoss(weight=turn_weight)

    def run(loader, train):
        net.train(train)
        tot = correct_turn = seen = 0
        key_hits = torch.zeros(5)
        with torch.set_grad_enabled(train):
            for xb, kb, tb in loader:
                xb, kb, tb = xb.to(dev), kb.to(dev), tb.to(dev)
                kl, tl = net(xb)
                loss = key_loss(kl, kb) + turn_loss(tl, tb)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                tot += loss.item() * len(xb)
                seen += len(xb)
                correct_turn += (tl.argmax(1) == tb).sum().item()
                key_hits += ((kl > 0).float() == kb).float().sum(0).cpu()
        return tot / seen, correct_turn / seen, (key_hits / seen).tolist()

    for ep in range(args.epochs):
        tl_, tt_, _ = run(tr, True)
        vl_, vt_, vk_ = run(va, False)
        kacc = " ".join(f"{v:.2f}" for v in vk_)
        print(
            f"[train] ep{ep + 1:2d} train_loss={tl_:.3f} val_loss={vl_:.3f} "
            f"val_turn_acc={vt_:.2f} val_key_acc(wasd,cr)=[{kacc}]"
        )

    # Save a checkpoint FIRST so a flaky export never costs us a training run.
    ckpt = os.path.splitext(args.out)[0] + ".pt"
    torch.save(net.state_dict(), ckpt)
    print(f"[train] saved checkpoint {ckpt}")
    try:
        export_onnx(net, args.out)
    except Exception as e:
        print(f"[train] ONNX export failed ({e}); checkpoint {ckpt} is safe.")
    print("[train] keys order = [w, a, s, d, crouch]; turn = [hardL, left, straight, right, hardR]")


if __name__ == "__main__":
    main()
