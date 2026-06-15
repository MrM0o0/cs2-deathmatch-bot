"""Behavioural-cloning movement policy: a frame in, movement commands out.

Loads the ONNX model trained by tools/train_movement.py and, each tick, predicts
which movement keys to hold and how hard to turn -- the learned-from-human
alternative to the hand-written waypoint navigator. Combat/aim are untouched;
this only drives roaming movement.
"""

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Mouse-x counts per turn class [hardL, left, straight, right, hardR].
# Magnitudes are config-tunable; the sign convention (which way is "left") is
# flippable via invert_turn, exactly like the waypoint navigator.
_TURN_SIGN = [-1, -1, 0, 1, 1]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class BCMovementPolicy:
    """Run the trained movement model and emit a command dict."""

    def __init__(
        self,
        model_path: str,
        mild_turn: int = 150,
        hard_turn: int = 400,
        invert_turn: bool = False,
        key_threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.mild_turn = mild_turn
        self.hard_turn = hard_turn
        self.invert_turn = invert_turn
        self.key_threshold = key_threshold
        self.session = None
        self.input_name = None

    def load(self) -> None:
        if ort is None:
            raise ImportError("onnxruntime not installed")
        providers = []
        available = ort.get_available_providers()
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[BCPolicy] Loaded {self.model_path} on {self.session.get_providers()[0]}")

    def act(self, frame: np.ndarray) -> dict:
        """Predict movement for the current BGR frame.

        Returns a command dict aligned with main.py's movement execution:
        forward/back/left/right/crouch (bool), turn_x (mouse counts), plus
        key_probs / turn_class for logging.
        """
        import cv2

        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        blob = small.astype(np.float32).transpose(2, 0, 1)[None] / 255.0  # 1x3x90x160 (BGR)
        keys_logits, turn_logits = self.session.run(None, {self.input_name: blob})
        probs = _sigmoid(keys_logits[0])  # [w, a, s, d, crouch]
        held = probs > self.key_threshold
        turn_class = int(np.argmax(turn_logits[0]))

        mag = self.hard_turn if turn_class in (0, 4) else self.mild_turn
        turn_x = _TURN_SIGN[turn_class] * mag
        if self.invert_turn:
            turn_x = -turn_x

        return {
            "forward": bool(held[0]),
            "left": bool(held[1]),
            "back": bool(held[2]),
            "right": bool(held[3]),
            "crouch": bool(held[4]),
            "turn_x": int(turn_x),
            "key_probs": [round(float(p), 2) for p in probs],
            "turn_class": turn_class,
        }
