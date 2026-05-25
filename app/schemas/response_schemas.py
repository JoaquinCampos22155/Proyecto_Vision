from pydantic import BaseModel, Field


class Thresholds(BaseModel):
    consistent: float
    needs_review: float
    main_view_low_similarity: float
    main_view_severe_similarity: float
    strong_color_mismatch: float


class ScoreBreakdown(BaseModel):
    raw_consistency_score: float
    main_view_score: float | None
    detail_support_score: float | None
    color_consistency_score: float
    robust_consistency_score: float
    low_confidence_reason: str | None = None


class ViewTypeEstimate(BaseModel):
    image_index: int
    view_type: str = Field(
        pattern="^(main_front|main_back|detail_collar|detail_logo|detail_label|detail_fabric|partial_view|invalid_or_low_quality|unknown)$"
    )
    confidence: float
    note: str


class FlaggedImage(BaseModel):
    image_index: int
    filename: str | None = None
    original_filename: str | None = None
    view_type: str
    severity: str = Field(pattern="^(low|medium|high)$")
    issue_type: str = "visual_outlier"
    reason: str
    recommended_action: str
    average_similarity_against_all: float | None = None
    average_similarity_against_same_view_type: float | None = None
    average_similarity_against_main_views: float | None = None


class DominantColor(BaseModel):
    image_index: int
    color_hex: str
    color_rgb: list[int]


class GarmentTypeEstimate(BaseModel):
    image_index: int
    label: str
    confidence: float
    note: str


class ConditionEstimate(BaseModel):
    label: str = Field(pattern="^(unknown|likely_new|likely_used)$")
    confidence: float
    note: str


class ImageDebugInfo(BaseModel):
    image_index: int
    original_filename: str
    detector_used: bool
    detection_found: bool
    detection_confidence: float | None
    bbox: list[int] | None
    fallback_used: bool
    crop_strategy: str
    rejected_detection_reason: str | None


class CropDebugUrl(BaseModel):
    image_index: int
    url: str


class ValidationResponse(BaseModel):
    product_id: str | None
    image_count: int
    status: str = Field(pattern="^(consistent|needs_review|inconsistent)$")
    scores: ScoreBreakdown
    note: str
    thresholds: Thresholds
    view_types: list[ViewTypeEstimate]
    flagged_images: list[FlaggedImage]
    quality_warnings: list[FlaggedImage] = []
    dominant_colors: list[DominantColor]
    garment_type_estimates: list[GarmentTypeEstimate]
    image_debug: list[ImageDebugInfo]
    crop_debug_urls: list[CropDebugUrl]
    pairwise_similarity_matrix: list[list[float]]
    condition_estimate: ConditionEstimate
