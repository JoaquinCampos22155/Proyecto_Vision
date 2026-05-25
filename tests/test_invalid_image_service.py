import numpy as np
from PIL import Image

from app.config import Settings
from app.services.invalid_image_service import InvalidImageService


def issue_types(findings):
    return {finding.issue_type for finding in findings}


def test_completely_dark_image_is_marked_too_dark():
    service = InvalidImageService(Settings())
    image = Image.new("RGB", (256, 256), (0, 0, 0))

    findings = service.analyze_image(image, 0, "dark.jpeg")

    assert "too_dark" in issue_types(findings)


def test_blurry_image_is_marked_blurry():
    service = InvalidImageService(Settings())
    image = Image.new("RGB", (256, 256), (128, 128, 128))

    findings = service.analyze_image(image, 0, "blur.jpeg")

    assert "blurry_image" in issue_types(findings)


def test_high_detail_normal_image_is_not_invalid_by_default():
    service = InvalidImageService(Settings())
    values = np.indices((256, 256)).sum(axis=0) % 2 * 255
    image = Image.fromarray(values.astype("uint8"), mode="L").convert("RGB")

    findings = service.analyze_image(image, 0, "normal.jpeg")

    assert findings == []
