from typing import List, Optional

from pydantic import BaseModel, Field


class SuspiciousImage(BaseModel):
    image: str
    average_similarity: float
    reason: str


class ImageSimilarityDiagnostic(BaseModel):
    image: str
    average_similarity: float
    min_similarity_against_others: float


class PairwiseSimilarity(BaseModel):
    image_a: str
    image_b: str
    similarity: float


class DominantColor(BaseModel):
    name: str
    hex: str
    percentage: float


class ImageColorDiagnostic(BaseModel):
    image: str
    dominant_colors: List[DominantColor]


class CropDiagnostic(BaseModel):
    image: str
    method: str
    bbox: Optional[List[int]] = None


class ConditionEstimation(BaseModel):
    estimated_condition: str
    confidence: float
    note: str


class ValidationResponse(BaseModel):
    product_id: str
    images_received: int

    consistency_score: float = Field(
        description="Score final de consistencia visual entre 0 y 1."
    )
    risk_level: str = Field(
        description="Nivel de riesgo: low, medium o high."
    )

    average_similarity: float
    lowest_similarity: float

    suspicious_images: List[SuspiciousImage]
    image_similarity_diagnostics: List[ImageSimilarityDiagnostic]
    pairwise_similarities: List[PairwiseSimilarity]

    global_dominant_colors: List[str]
    image_color_diagnostics: List[ImageColorDiagnostic]

    condition_estimation: ConditionEstimation
    crop_diagnostics: List[CropDiagnostic]

    message: str