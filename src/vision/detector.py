"""YOLO ONNX inference for enemy detection."""

import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import cv2
except ImportError:
    cv2 = None


class Detection:
    """A single detected object."""

    __slots__ = ("class_id", "class_name", "confidence", "x1", "y1", "x2", "y2")

    def __init__(self, class_id: int, class_name: str, confidence: float,
                 x1: float, y1: float, x2: float, y2: float):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_head(self) -> bool:
        return "head" in self.class_name

    def __repr__(self) -> str:
        cx, cy = self.center
        return (f"Detection({self.class_name} {self.confidence:.2f} "
                f"@ ({cx:.0f}, {cy:.0f}))")


class YOLODetector:
    """ONNX-based YOLO object detector with DirectML support."""

    def __init__(self, model_path: str, input_size: int = 640,
                 confidence_threshold: float = 0.45,
                 nms_threshold: float = 0.5,
                 classes: list[str] | None = None):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thresh = confidence_threshold
        self.nms_thresh = nms_threshold
        self.classes = classes or ["ct_player", "head"]
        self.session = None
        self.input_name = None
        self._inference_ms = 0.0

    def load(self) -> None:
        """Load the ONNX model. Tries DirectML, falls back to CPU."""
        if ort is None:
            raise ImportError("onnxruntime not installed")

        providers = []
        available = ort.get_available_providers()

        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        active = self.session.get_providers()
        print(f"[Detector] Loaded {self.model_path} on {active[0]}")

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Preprocess BGR frame for YOLO input.

        Returns:
            (blob, scale, pad_x, pad_y)
        """
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # HWC BGR -> CHW RGB, normalize to [0, 1]
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)  # Add batch dim

        return blob, scale, pad_x, pad_y

    def postprocess(self, output: np.ndarray, scale: float,
                    pad_x: float, pad_y: float) -> list[Detection]:
        """Parse YOLO output into detections.

        Handles YOLOv8 output format: (1, 4+num_classes, num_boxes)
        """
        # YOLOv8 outputs (1, 4+C, N) - transpose to (N, 4+C)
        predictions = output[0].T  # (N, 4+C)

        # Filter by confidence
        class_scores = predictions[:, 4:]  # (N, C)
        max_scores = class_scores.max(axis=1)
        mask = max_scores > self.conf_thresh
        predictions = predictions[mask]
        class_scores = class_scores[mask]
        max_scores = max_scores[mask]

        if len(predictions) == 0:
            return []

        class_ids = class_scores.argmax(axis=1)

        # Convert cx, cy, w, h -> x1, y1, x2, y2
        cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Remove padding and rescale to original frame coordinates
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # NMS
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        indices = self._nms(boxes, max_scores)

        detections = []
        for i in indices:
            cid = int(class_ids[i])
            det = Detection(
                class_id=cid,
                class_name=self.classes[cid] if cid < len(self.classes) else f"class_{cid}",
                confidence=float(max_scores[i]),
                x1=float(x1[i]),
                y1=float(y1[i]),
                x2=float(x2[i]),
                y2=float(y2[i]),
            )
            detections.append(det)

        return detections

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        """Non-maximum suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            remaining = np.where(iou <= self.nms_thresh)[0]
            order = order[remaining + 1]

        return keep

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run full detection pipeline on a BGR frame."""
        if self.session is None:
            return []

        t0 = time.perf_counter()
        blob, scale, pad_x, pad_y = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: blob})[0]
        detections = self.postprocess(output, scale, pad_x, pad_y)
        self._inference_ms = (time.perf_counter() - t0) * 1000

        return detections

    @property
    def inference_ms(self) -> float:
        return self._inference_ms
