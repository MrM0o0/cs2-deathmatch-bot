"""View YOLO annotations overlaid on images for verification."""

import os
import sys
import argparse

try:
    import cv2
except ImportError:
    print("OpenCV required: pip install opencv-python")
    sys.exit(1)


CLASSES = ["ct", "t", "head_ct", "head_t"]
COLORS = [
    (255, 150, 50),   # ct - blue
    (50, 50, 255),    # t - red
    (255, 200, 100),  # head_ct - light blue
    (100, 100, 255),  # head_t - light red
]


def view_labels(image_dir: str, label_dir: str):
    """Display images with their YOLO label overlays.

    Args:
        image_dir: Directory of images.
        label_dir: Directory of YOLO format label files.
    """
    images = sorted([f for f in os.listdir(image_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(images)} images. Arrow keys to navigate, ESC to quit.")

    idx = 0
    while True:
        img_name = images[idx]
        img_path = os.path.join(image_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        img = cv2.imread(img_path)
        if img is None:
            idx = (idx + 1) % len(images)
            continue

        h, w = img.shape[:2]

        # Draw labels if they exist
        label_count = 0
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # Convert YOLO format to pixel coords
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    color = COLORS[cls_id] if cls_id < len(COLORS) else (200, 200, 200)
                    cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls_{cls_id}"

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, cls_name, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    label_count += 1

        # Info text
        info = f"[{idx + 1}/{len(images)}] {img_name} - {label_count} labels"
        cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0), 2)

        cv2.imshow("Label Viewer", img)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord("d") or key == 83:  # Right arrow
            idx = (idx + 1) % len(images)
        elif key == ord("a") or key == 81:  # Left arrow
            idx = (idx - 1) % len(images)
        elif key == ord("x"):  # Delete label
            if os.path.exists(label_path):
                os.remove(label_path)
                print(f"Deleted {label_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Label Viewer")
    parser.add_argument("--images", "-i", default="models/training/dataset/images")
    parser.add_argument("--labels", "-l", default="models/training/dataset/labels")
    args = parser.parse_args()

    view_labels(args.images, args.labels)
