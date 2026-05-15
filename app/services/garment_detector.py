from PIL import Image

from app.config import Settings
from app.services.validation_service_types import DetectionResult
from app.utils.image_utils import clamp_bbox


class GarmentDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            # Replace this path with a fine-tuned garment detector when product data is available.
            self._model = YOLO(self.settings.yolo_model_path)
        return self._model

    def detect_primary_garment(self, image: Image.Image) -> DetectionResult:
        try:
            results = self.model.predict(image, verbose=False, conf=self.settings.yolo_confidence_threshold)
        except Exception:
            return DetectionResult(
                bbox_xyxy=None,
                class_name=None,
                confidence=None,
                detector_used=False,
                detection_found=False,
                used_fallback=True,
                rejected_detection_reason="YOLO model could not run.",
            )

        if not results:
            return DetectionResult(
                bbox_xyxy=None,
                class_name=None,
                confidence=None,
                detector_used=True,
                detection_found=False,
                used_fallback=True,
                rejected_detection_reason="YOLO returned no results.",
            )

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return DetectionResult(
                bbox_xyxy=None,
                class_name=None,
                confidence=None,
                detector_used=True,
                detection_found=False,
                used_fallback=True,
                rejected_detection_reason="YOLO returned no bounding boxes.",
            )

        names = getattr(result, "names", {}) or {}
        garment_names = {name.lower() for name in self.settings.garment_class_names}
        rejected_names = {name.lower() for name in self.settings.yolo_rejected_class_names}
        candidates = []
        rejection_reasons = []

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().tolist()
            confidence = float(box.conf[0].detach().cpu().item())
            class_id = int(box.cls[0].detach().cpu().item())
            class_name = str(names.get(class_id, f"class_{class_id}")).lower()
            x1, y1, x2, y2 = clamp_bbox(tuple(round(value) for value in xyxy), image.size)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            image_area = image.size[0] * image.size[1]
            area_ratio = area / image_area if image_area else 0.0
            center_x = ((x1 + x2) / 2) / image.size[0] if image.size[0] else 0.0

            if confidence < self.settings.yolo_min_accepted_confidence:
                rejection_reasons.append(f"{class_name} rejected: confidence {confidence:.2f} < minimum.")
                continue

            if class_name in rejected_names:
                rejection_reasons.append(f"{class_name} rejected: class is not useful for garment crops.")
                continue

            if class_name not in garment_names:
                rejection_reasons.append(f"{class_name} rejected: not in garment class allowlist.")
                continue

            if area_ratio < self.settings.yolo_min_bbox_area_ratio:
                rejection_reasons.append(f"{class_name} rejected: bbox covers only {area_ratio:.2f} of image.")
                continue

            if center_x < self.settings.yolo_min_bbox_center_x or center_x > self.settings.yolo_max_bbox_center_x:
                rejection_reasons.append(f"{class_name} rejected: bbox is too lateral.")
                continue

            candidates.append((confidence * max(area, 1), (x1, y1, x2, y2), class_name, confidence))

        if not candidates:
            return DetectionResult(
                bbox_xyxy=None,
                class_name=None,
                confidence=None,
                detector_used=True,
                detection_found=False,
                used_fallback=True,
                rejected_detection_reason=" ".join(rejection_reasons) or "No accepted garment detection.",
            )

        _, bbox, class_name, confidence = max(candidates, key=lambda item: item[0])
        return DetectionResult(
            bbox_xyxy=bbox,
            class_name=class_name,
            confidence=confidence,
            detector_used=True,
            detection_found=True,
            used_fallback=False,
            rejected_detection_reason=None,
        )
