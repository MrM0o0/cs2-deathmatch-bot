"""ROI extraction from captured frames (HUD, crosshair area, minimap)."""

import numpy as np


def extract_region(frame: np.ndarray, region: list[int] | tuple[int, ...]) -> np.ndarray:
    """Extract a rectangular region from a frame.

    Args:
        frame: Full BGR frame (H, W, 3).
        region: (x, y, w, h) rectangle.

    Returns:
        Cropped BGR region.
    """
    x, y, w, h = region
    return frame[y:y + h, x:x + w].copy()


def extract_crosshair_area(frame: np.ndarray, cx: int, cy: int,
                           size: int = 200) -> np.ndarray:
    """Extract the area around the crosshair for close-range detection."""
    half = size // 2
    h, w = frame.shape[:2]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return frame[y1:y2, x1:x2].copy()


def extract_minimap(frame: np.ndarray, x: int, y: int,
                    size: int) -> np.ndarray:
    """Extract the minimap region."""
    return frame[y:y + size, x:x + size].copy()
