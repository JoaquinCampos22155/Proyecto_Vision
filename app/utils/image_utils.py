from PIL import Image


def clamp_bbox(bbox_xyxy: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def crop_or_original(image: Image.Image, bbox_xyxy: tuple[int, int, int, int] | None) -> Image.Image:
    if bbox_xyxy is None:
        return image
    return image.crop(clamp_bbox(bbox_xyxy, image.size))


def center_crop(image: Image.Image, crop_ratio: float = 0.82) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * crop_ratio))
    crop_height = max(1, round(height * crop_ratio))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


def crop_by_strategy(image: Image.Image, strategy: str) -> Image.Image:
    if strategy == "center_crop":
        return center_crop(image)
    return image
