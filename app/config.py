from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    min_images: int = Field(default=6, ge=1)
    max_images: int = Field(default=10, ge=1)

    # consistent_threshold default origin 0.7
    consistent_threshold: float = Field(default=0.80, ge=-1.0, le=1.0)
    review_threshold: float = Field(default=0.50, ge=-1.0, le=1.0)

    # main_view_weight default origin 0.65
    main_view_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    # color_consistency_weight default origin 0.25
    color_consistency_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    detail_support_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    # main_view_low_similarity_threshold default origin 0.45
    main_view_low_similarity_threshold: float = Field(default=0.65, ge=-1.0, le=1.0)
    # main_view_severe_similarity_threshold default origin 0.35
    main_view_severe_similarity_threshold: float = Field(default=0.50, ge=-1.0, le=1.0)
    detail_visual_support_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)
    # strong_color_mismatch_threshold default origin 0.55
    strong_color_mismatch_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
