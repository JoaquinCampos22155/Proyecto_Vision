from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class LoadedImage:
    index: int
    filename: str
    image: Image.Image


@dataclass(frozen=True)
class DetectionResult:
    bbox_xyxy: tuple[int, int, int, int] | None
    class_name: str | None
    confidence: float | None
    detector_used: bool
    detection_found: bool
    used_fallback: bool
    rejected_detection_reason: str | None = None
