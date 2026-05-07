from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


def load_image_from_upload_bytes(image_bytes: bytes, filename: str) -> Image.Image:
    """
    Convierte bytes de una imagen subida a un objeto PIL RGB.
    """

    if not image_bytes:
        raise ValueError(f"The file {filename} is empty.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image = image.convert("RGB")
        return image

    except UnidentifiedImageError:
        raise ValueError(f"The file {filename} is not a valid image.")


def center_crop(image: Image.Image, crop_ratio: float = 0.82) -> Image.Image:
    """
    Hace un recorte central como fallback cuando YOLO no logra detectar un objeto útil.
    """

    width, height = image.size

    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)

    left = int((width - new_width) / 2)
    top = int((height - new_height) / 2)
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom))


def crop_with_bbox(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Recorta una imagen usando bounding box.
    """

    width, height = image.size
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return center_crop(image)

    return image.crop((x1, y1, x2, y2))


def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    """
    Convierte PIL Image RGB a numpy array RGB.
    """

    return np.array(image.convert("RGB"))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convierte RGB a HEX.
    """

    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"