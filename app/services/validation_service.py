from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import Settings
from app.schemas.response_schemas import (
    CropDebugUrl,
    DominantColor,
    FlaggedImage,
    GarmentTypeEstimate,
    ImageDebugInfo,
    Thresholds,
    ValidationResponse,
    ViewTypeEstimate,
)
from app.services.color_service import ColorService
from app.services.condition_service import ConditionService
from app.services.embedding_service import EmbeddingService
from app.services.garment_detector import GarmentDetector
from app.services.image_loader import ImageLoader, ImageLoadingError
from app.services.invalid_image_service import InvalidImageFinding, InvalidImageService
from app.services.scoring_service import ScoringService
from app.services.similarity_service import SimilarityService
from app.services.validation_service_types import DetectionResult
from app.services.view_type_service import DETAIL_VIEW_TYPES, ViewTypeService
from app.utils.image_utils import crop_by_strategy, crop_or_original


DEBUG_CROP_DIR = Path(__file__).resolve().parents[1] / "static" / "debug_crops"
MVP_CALIBRATION_NOTE = (
    "This is an MVP compatibility score using generic pretrained embeddings. It should be calibrated with real "
    "ecommerce image datasets before production use."
)


class ValidationError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ProductImageValidationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.image_loader = ImageLoader()
        self.detector = GarmentDetector(settings=settings)
        self.embedding_service = EmbeddingService(settings=settings)
        self.similarity_service = SimilarityService()
        self.scoring_service = ScoringService(settings=settings)
        self.color_service = ColorService()
        self.condition_service = ConditionService()
        self.view_type_service = ViewTypeService()
        self.invalid_image_service = InvalidImageService(settings=settings)

    async def validate(self, product_id: str | None, uploads: list[UploadFile]) -> ValidationResponse:
        if len(uploads) < self.settings.min_images:
            raise ValidationError(f"At least {self.settings.min_images} images are required.")
        if len(uploads) > self.settings.max_images:
            raise ValidationError(f"No more than {self.settings.max_images} images are allowed.")

        loaded_images = []
        for index, upload in enumerate(uploads):
            try:
                loaded_images.append(await self.image_loader.load(upload, image_index=index))
            except ImageLoadingError as exc:
                raise ValidationError(exc.message) from exc

        cropped_images = []
        dominant_colors: list[DominantColor] = []
        dominant_color_rgbs: list[tuple[int, int, int]] = []
        garment_type_estimates: list[GarmentTypeEstimate] = []
        image_debug: list[ImageDebugInfo] = []
        crop_debug_urls: list[CropDebugUrl] = []
        view_type_results = []

        for loaded in loaded_images:
            detection, cropped, crop_strategy = self._prepare_embedding_image(loaded.image)
            cropped_images.append(cropped)
            view_type_results.append(
                self.view_type_service.estimate(cropped, image_index=loaded.index, filename=loaded.filename)
            )
            crop_debug_urls.append(
                CropDebugUrl(
                    image_index=loaded.index,
                    url=self._save_debug_crop(cropped, image_index=loaded.index, product_id=product_id),
                )
            )

            color_rgb = self.color_service.dominant_color_rgb(cropped)
            dominant_color_rgbs.append(color_rgb)
            dominant_colors.append(
                DominantColor(
                    image_index=loaded.index,
                    color_hex=f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}",
                    color_rgb=list(color_rgb),
                )
            )
            garment_type_estimates.append(self._garment_estimate_from_detection(loaded.index, detection))
            image_debug.append(
                ImageDebugInfo(
                    image_index=loaded.index,
                    original_filename=loaded.filename,
                    detector_used=detection.detector_used,
                    detection_found=detection.detection_found,
                    detection_confidence=round(float(detection.confidence), 4)
                    if detection.confidence is not None
                    else None,
                    bbox=list(detection.bbox_xyxy) if detection.bbox_xyxy else None,
                    fallback_used=detection.used_fallback,
                    crop_strategy=crop_strategy,
                    rejected_detection_reason=detection.rejected_detection_reason,
                )
            )

        embeddings = self.embedding_service.extract_embeddings(cropped_images)
        similarity_matrix = self.similarity_service.cosine_similarity_matrix(embeddings)
        scoring_result = self.scoring_service.score(
            similarity_matrix=similarity_matrix,
            view_types=view_type_results,
            dominant_color_rgbs=dominant_color_rgbs,
        )
        invalid_findings = self.invalid_image_service.analyze_images(
            [loaded.image for loaded in loaded_images],
            [loaded.filename for loaded in loaded_images],
        )
        invalid_findings, quality_warning_findings = self._split_invalid_findings(
            invalid_findings,
            view_type_results,
            scoring_result.scores.main_view_score,
            scoring_result.scores.color_consistency_score,
            scoring_result.scores.robust_consistency_score,
        )
        flagged_images = self._merge_flagged_images(
            scoring_result.flagged_images,
            [self._flag_from_invalid_finding(finding) for finding in invalid_findings],
            [loaded.filename for loaded in loaded_images],
        )
        quality_warnings = self._merge_flagged_images(
            [],
            [self._flag_from_invalid_finding(finding) for finding in quality_warning_findings],
            [loaded.filename for loaded in loaded_images],
        )
        final_status = self.scoring_service.resolve_status(
            robust_score=scoring_result.scores.robust_consistency_score,
            flagged_images=flagged_images,
            view_types=view_type_results,
            dominant_color_rgbs=dominant_color_rgbs,
        )
        if final_status == "consistent" and scoring_result.status == "needs_review":
            final_status = "needs_review"

        return ValidationResponse(
            product_id=product_id,
            image_count=len(uploads),
            status=final_status,
            scores=scoring_result.scores,
            note=MVP_CALIBRATION_NOTE,
            thresholds=Thresholds(
                consistent=self.settings.consistent_threshold,
                needs_review=self.settings.review_threshold,
                main_view_low_similarity=self.settings.main_view_low_similarity_threshold,
                main_view_severe_similarity=self.settings.main_view_severe_similarity_threshold,
                strong_color_mismatch=self.settings.strong_color_mismatch_threshold,
            ),
            view_types=[
                ViewTypeEstimate(
                    image_index=item.image_index,
                    view_type=item.view_type,
                    confidence=round(item.confidence, 4),
                    note=item.note,
                )
                for item in view_type_results
            ],
            flagged_images=flagged_images,
            quality_warnings=quality_warnings,
            dominant_colors=dominant_colors,
            garment_type_estimates=garment_type_estimates,
            image_debug=image_debug,
            crop_debug_urls=crop_debug_urls,
            pairwise_similarity_matrix=self.similarity_service.rounded_matrix(similarity_matrix),
            condition_estimate=self.condition_service.estimate_condition(),
        )

    def _prepare_embedding_image(self, image) -> tuple[DetectionResult, object, str]:
        if not self.settings.use_yolo_crops:
            return (
                DetectionResult(
                    bbox_xyxy=None,
                    class_name=None,
                    confidence=None,
                    detector_used=False,
                    detection_found=False,
                    used_fallback=True,
                    rejected_detection_reason=None,
                ),
                crop_by_strategy(image, self.settings.default_crop_strategy),
                self.settings.default_crop_strategy,
            )

        detection = self.detector.detect_primary_garment(image)
        if detection.detection_found and detection.bbox_xyxy:
            return detection, crop_or_original(image, detection.bbox_xyxy), "yolo_bbox"

        return detection, crop_by_strategy(image, self.settings.default_crop_strategy), self.settings.default_crop_strategy

    @staticmethod
    def _garment_estimate_from_detection(image_index: int, detection) -> GarmentTypeEstimate:
        if detection.class_name and not detection.used_fallback:
            return GarmentTypeEstimate(
                image_index=image_index,
                label=detection.class_name,
                confidence=round(float(detection.confidence), 4),
                note="Auxiliary YOLO estimate. It is not used as the primary consistency decision.",
            )

        return GarmentTypeEstimate(
            image_index=image_index,
            label="unknown",
            confidence=0.0,
            note="No garment-specific detection was found. The full image or a generic YOLO crop was used as fallback.",
        )

    @staticmethod
    def _save_debug_crop(cropped_image, image_index: int, product_id: str | None) -> str:
        DEBUG_CROP_DIR.mkdir(parents=True, exist_ok=True)
        safe_product_id = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in (product_id or "product")
        )
        filename = f"{safe_product_id}_{uuid4().hex}_image_{image_index}.jpg"
        output_path = DEBUG_CROP_DIR / filename
        cropped_image.convert("RGB").save(output_path, format="JPEG", quality=90)
        return f"/static/debug_crops/{filename}"

    @staticmethod
    def _flag_from_invalid_finding(finding: InvalidImageFinding) -> FlaggedImage:
        return FlaggedImage(
            image_index=finding.image_index,
            filename=finding.filename,
            original_filename=finding.filename,
            view_type="invalid_or_low_quality",
            severity=finding.severity,
            issue_type=finding.issue_type,
            reason=finding.reason,
            recommended_action=finding.recommended_action,
        )

    @staticmethod
    def _merge_flagged_images(
        scoring_flags: list[FlaggedImage],
        invalid_flags: list[FlaggedImage],
        ordered_filenames: list[str],
    ) -> list[FlaggedImage]:
        merged: list[FlaggedImage] = []
        seen: set[tuple[int, str]] = set()
        for flag in [*scoring_flags, *invalid_flags]:
            if flag.filename is None and 0 <= flag.image_index < len(ordered_filenames):
                flag.filename = ordered_filenames[flag.image_index]
            if flag.original_filename is None and 0 <= flag.image_index < len(ordered_filenames):
                flag.original_filename = ordered_filenames[flag.image_index]
            key = (flag.image_index, flag.issue_type)
            if key in seen:
                continue
            seen.add(key)
            merged.append(flag)
        return merged

    def _split_invalid_findings(
        self,
        findings: list[InvalidImageFinding],
        view_type_results,
        main_view_score: float | None,
        color_consistency_score: float,
        robust_consistency_score: float,
    ) -> tuple[list[InvalidImageFinding], list[InvalidImageFinding]]:
        view_type_by_index = {item.image_index: item.view_type for item in view_type_results}
        main_and_color_are_healthy = (
            main_view_score is not None
            and main_view_score >= self.settings.consistent_threshold
            and color_consistency_score >= self.settings.consistent_threshold
        )

        strong_product_signal = (
            robust_consistency_score >= 0.75
            and color_consistency_score >= 0.90
            and main_view_score is not None
            and main_view_score >= 0.65
        )
        blur_only_findings = (
            findings
            and all(finding.issue_type == "blurry_image" and finding.severity == "medium" for finding in findings)
        )
        if strong_product_signal and blur_only_findings:
            return [], findings

        if not main_and_color_are_healthy:
            return findings, []

        single_medium_blur_with_strong_product_signal = (
            len(findings) == 1
            and findings[0].issue_type == "blurry_image"
            and findings[0].severity == "medium"
            and robust_consistency_score >= 0.80
            and color_consistency_score >= 0.90
            and main_view_score is not None
            and main_view_score >= 0.70
        )
        if single_medium_blur_with_strong_product_signal:
            return [], findings

        filtered = []
        warnings = []
        for finding in findings:
            is_detail_blur = (
                finding.issue_type == "blurry_image"
                and view_type_by_index.get(finding.image_index) in DETAIL_VIEW_TYPES
            )
            if is_detail_blur:
                warnings.append(finding)
            else:
                filtered.append(finding)
        return filtered, warnings
