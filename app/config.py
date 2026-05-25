from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    min_images: int = Field(default=6, ge=1)
    max_images: int = Field(default=10, ge=1)

    consistent_threshold: float = Field(default=0.70, ge=-1.0, le=1.0)
    review_threshold: float = Field(default=0.50, ge=-1.0, le=1.0)

    main_view_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    color_consistency_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    detail_support_weight: float = Field(default=0.05, ge=0.0, le=1.0)

    main_view_low_similarity_threshold: float = Field(default=0.45, ge=-1.0, le=1.0)
    main_view_severe_similarity_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)
    detail_visual_support_threshold: float = Field(default=0.25, ge=-1.0, le=1.0)
    strong_color_mismatch_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    detail_color_mismatch_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    detail_outlier_count_for_review: int = Field(default=2, ge=1)
    high_main_flags_for_inconsistent: int = Field(default=2, ge=1)
    invalid_images_for_inconsistent: int = Field(default=3, ge=1)
    color_cluster_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    consensus_cluster_low_support_threshold: float = Field(default=0.50, ge=-1.0, le=1.0)
    consensus_cluster_gap_threshold: float = Field(default=0.18, ge=0.0, le=2.0)
    model_cluster_within_similarity_threshold: float = Field(default=0.88, ge=-1.0, le=1.0)
    model_cluster_between_similarity_threshold: float = Field(default=0.58, ge=-1.0, le=1.0)
    model_cluster_color_similarity_threshold: float = Field(default=0.90, ge=0.0, le=1.0)

    use_yolo_crops: bool = False
    default_crop_strategy: Literal["full_image", "center_crop"] = "full_image"

    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_min_accepted_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    yolo_min_bbox_area_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    yolo_min_bbox_center_x: float = Field(default=0.20, ge=0.0, le=1.0)
    yolo_max_bbox_center_x: float = Field(default=0.80, ge=0.0, le=1.0)
    yolo_rejected_class_names: tuple[str, ...] = ("tie",)
    garment_class_names: tuple[str, ...] = (
        "shirt",
        "t-shirt",
        "pants",
        "jeans",
        "dress",
        "skirt",
        "shorts",
        "jacket",
        "coat",
        "hoodie",
        "sweater",
        "shoe",
        "hat",
        "bag",
        "handbag",
        "backpack",
    )

    resnet_model_name: Literal["resnet18", "resnet50"] = "resnet50"
    device: str = "auto"

    invalid_dark_brightness_threshold: float = Field(default=35.0, ge=0.0, le=255.0)
    invalid_blur_laplacian_threshold: float = Field(default=25.0, ge=0.0)
    invalid_min_resolution_px: int = Field(default=224, ge=1)
    invalid_duplicate_hamming_threshold: int = Field(default=1, ge=0)
    invalid_duplicate_mean_abs_diff_threshold: float = Field(default=2.0, ge=0.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
