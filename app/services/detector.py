import os
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from app.utils.image_utils import center_crop, crop_with_bbox


class GarmentDetector:
    """
    Detector inicial para aislar la región principal del producto.

    En Fase 1:
    - Intenta usar YOLO preentrenado.
    - Si YOLO no detecta nada útil, usa crop central.
    - No hace fine-tuning todavía.
    """

    def __init__(self):
        self.enable_yolo = os.getenv("ENABLE_YOLO", "true").lower() == "true"
        self.model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
        self.confidence_threshold = float(os.getenv("YOLO_CONFIDENCE", "0.25"))

        self.model = None

        if self.enable_yolo:
            self._load_yolo_model()

    def _load_yolo_model(self):
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_name)

        except Exception:
            self.model = None
            self.enable_yolo = False

    def crop_product(self, image: Image.Image) -> Dict:
        """
        Devuelve:
        {
            "image": cropped_image,
            "method": "yolo" o "center_crop",
            "bbox": [x1, y1, x2, y2] o None
        }
        """

        if not self.enable_yolo or self.model is None:
            return {
                "image": center_crop(image),
                "method": "center_crop",
                "bbox": None,
            }

        try:
            image_np = np.array(image.convert("RGB"))

            results = self.model.predict(
                source=image_np,
                conf=self.confidence_threshold,
                verbose=False,
            )

            if not results:
                return self._fallback_crop(image)

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return self._fallback_crop(image)

            boxes = result.boxes.xyxy.cpu().numpy()

            largest_box = self._get_largest_box(boxes)

            if largest_box is None:
                return self._fallback_crop(image)

            x1, y1, x2, y2 = largest_box
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            cropped_image = crop_with_bbox(
                image=image,
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            )

            return {
                "image": cropped_image,
                "method": "yolo",
                "bbox": bbox,
            }

        except Exception:
            return self._fallback_crop(image)

    def _fallback_crop(self, image: Image.Image) -> Dict:
        return {
            "image": center_crop(image),
            "method": "center_crop",
            "bbox": None,
        }

    def _get_largest_box(self, boxes: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Selecciona el bounding box con mayor área.
        """

        if boxes is None or len(boxes) == 0:
            return None

        largest_area = 0
        largest_box = None

        for box in boxes:
            x1, y1, x2, y2 = box
            area = max(0, x2 - x1) * max(0, y2 - y1)

            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2)

        return largest_box