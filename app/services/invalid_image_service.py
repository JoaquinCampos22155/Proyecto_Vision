from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from PIL import Image

from app.config import Settings, get_settings


@dataclass(frozen=True)
class InvalidImageFinding:
    image_index: int
    filename: str
    issue_type: str
    severity: str
    reason: str
    recommended_action: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class InvalidImageService:
    """Conservative MVP checks for low-quality images.

    Findings are quality evidence, not product-mismatch evidence. The validation
    pipeline decides whether each finding is blocking or only a non-blocking
    quality warning based on the product-consistency signal.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def analyze_image(self, image: Image.Image, image_index: int, filename: str) -> list[InvalidImageFinding]:
        rgb_image = image.convert("RGB")
        findings: list[InvalidImageFinding] = []

        if self._is_low_resolution(rgb_image):
            findings.append(
                InvalidImageFinding(
                    image_index=image_index,
                    filename=filename,
                    issue_type="low_resolution",
                    severity="medium",
                    reason="The image resolution is too low for reliable visual validation.",
                    recommended_action="Request a clearer image with higher resolution.",
                )
            )

        brightness = self._average_brightness(rgb_image)
        if brightness < self.settings.invalid_dark_brightness_threshold:
            findings.append(
                InvalidImageFinding(
                    image_index=image_index,
                    filename=filename,
                    issue_type="too_dark",
                    severity="medium",
                    reason="The image is too dark to reliably validate product consistency.",
                    recommended_action="Review manually or request a brighter image.",
                )
            )
            return findings

        blur_score = self._laplacian_variance(rgb_image)
        if blur_score < self.settings.invalid_blur_laplacian_threshold:
            findings.append(
                InvalidImageFinding(
                    image_index=image_index,
                    filename=filename,
                    issue_type="blurry_image",
                    severity="medium",
                    reason="The image appears blurry and may not provide reliable visual evidence.",
                    recommended_action="Review manually or request a sharper image.",
                )
            )

        return findings

    def analyze_images(self, images: Iterable[Image.Image], filenames: Iterable[str]) -> list[InvalidImageFinding]:
        image_list = list(images)
        filename_list = list(filenames)
        findings: list[InvalidImageFinding] = []

        for index, image in enumerate(image_list):
            filename = filename_list[index] if index < len(filename_list) else f"image_{index}"
            findings.extend(self.analyze_image(image, index, filename))

        findings.extend(self._duplicate_findings(image_list, filename_list))
        return findings

    def _is_low_resolution(self, image: Image.Image) -> bool:
        min_size = self.settings.invalid_min_resolution_px
        width, height = image.size
        return width < min_size or height < min_size

    def _average_brightness(self, image: Image.Image) -> float:
        grayscale = np.asarray(image.convert("L"), dtype=np.float32)
        return float(grayscale.mean())

    def _laplacian_variance(self, image: Image.Image) -> float:
        grayscale = np.asarray(image.convert("L"), dtype=np.float32)
        padded = np.pad(grayscale, 1, mode="edge")
        laplacian = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4 * padded[1:-1, 1:-1]
        )
        return float(laplacian.var())

    def _average_hash(self, image: Image.Image) -> np.ndarray:
        small = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        values = np.asarray(small, dtype=np.float32)
        return values > values.mean()

    def _mean_abs_difference(self, left: Image.Image, right: Image.Image) -> float:
        left_array = np.asarray(left.convert("RGB").resize((64, 64)), dtype=np.float32)
        right_array = np.asarray(right.convert("RGB").resize((64, 64)), dtype=np.float32)
        return float(np.abs(left_array - right_array).mean())

    def _duplicate_findings(self, images: list[Image.Image], filenames: list[str]) -> list[InvalidImageFinding]:
        findings: list[InvalidImageFinding] = []
        hashes = [self._average_hash(image) for image in images]
        duplicate_indexes: set[int] = set()

        for index in range(len(hashes)):
            for other_index in range(index + 1, len(hashes)):
                distance = int(np.count_nonzero(hashes[index] != hashes[other_index]))
                mean_abs_diff = self._mean_abs_difference(images[index], images[other_index])
                if (
                    distance <= self.settings.invalid_duplicate_hamming_threshold
                    and mean_abs_diff <= self.settings.invalid_duplicate_mean_abs_diff_threshold
                ):
                    duplicate_indexes.add(other_index)

        for index in sorted(duplicate_indexes):
            filename = filenames[index] if index < len(filenames) else f"image_{index}"
            findings.append(
                InvalidImageFinding(
                    image_index=index,
                    filename=filename,
                    issue_type="duplicate_image",
                    severity="low",
                    reason="The image appears to be a duplicate or near-duplicate of another image in the set.",
                    recommended_action="Review manually; duplicates may be acceptable but add little validation value.",
                )
            )
        return findings
