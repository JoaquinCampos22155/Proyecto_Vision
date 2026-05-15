from dataclasses import dataclass
from statistics import median

import torch

from app.config import Settings
from app.schemas.response_schemas import FlaggedImage, ScoreBreakdown
from app.services.color_service import ColorService
from app.services.similarity_service import SimilarityService
from app.services.view_type_service import DETAIL_VIEW_TYPES, MAIN_VIEW_TYPES, ViewTypeResult


@dataclass(frozen=True)
class CompatibilityScoringResult:
    scores: ScoreBreakdown
    flagged_images: list[FlaggedImage]
    status: str


class ScoringService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def score(
        self,
        similarity_matrix: torch.Tensor,
        view_types: list[ViewTypeResult],
        dominant_color_rgbs: list[tuple[int, int, int]],
    ) -> CompatibilityScoringResult:
        raw_score = SimilarityService.consistency_score(similarity_matrix)
        main_indices = [
            estimate.image_index for estimate in view_types if estimate.view_type in MAIN_VIEW_TYPES
        ]
        detail_indices = [
            estimate.image_index for estimate in view_types if estimate.view_type in DETAIL_VIEW_TYPES
        ]

        main_view_score = self._score_indices(similarity_matrix, main_indices)
        detail_support_score = self._detail_support_score(similarity_matrix, detail_indices, main_indices)
        color_consistency_score = self._color_consistency_score(dominant_color_rgbs, main_indices)

        low_confidence_reason = None
        if main_view_score is None:
            low_confidence_reason = "Fewer than two main or partial views were detected; raw score is used as fallback."

        robust_score = self._weighted_score(
            main_score=main_view_score if main_view_score is not None else raw_score,
            color_score=color_consistency_score,
            detail_score=detail_support_score,
        )

        averages = self._average_similarity_details(similarity_matrix, view_types, main_indices)
        flagged_images = self._flag_images(
            view_types=view_types,
            averages=averages,
            dominant_color_rgbs=dominant_color_rgbs,
            main_indices=main_indices,
        )
        status = self._status(
            robust_score=robust_score,
            flagged_images=flagged_images,
            main_indices=main_indices,
            dominant_color_rgbs=dominant_color_rgbs,
        )

        return CompatibilityScoringResult(
            scores=ScoreBreakdown(
                raw_consistency_score=round(raw_score, 4),
                main_view_score=round(main_view_score, 4) if main_view_score is not None else None,
                detail_support_score=round(detail_support_score, 4) if detail_support_score is not None else None,
                color_consistency_score=round(color_consistency_score, 4),
                robust_consistency_score=round(robust_score, 4),
                low_confidence_reason=low_confidence_reason,
            ),
            flagged_images=flagged_images,
            status=status,
        )

    @staticmethod
    def _score_indices(similarity_matrix: torch.Tensor, indices: list[int]) -> float | None:
        if len(indices) < 2:
            return None
        values = [
            float(similarity_matrix[left, right].item())
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
        ]
        return float(median(values))

    @staticmethod
    def _detail_support_score(
        similarity_matrix: torch.Tensor, detail_indices: list[int], main_indices: list[int]
    ) -> float | None:
        if not detail_indices:
            return None

        values = []
        if main_indices:
            for detail_index in detail_indices:
                values.append(max(float(similarity_matrix[detail_index, main_index].item()) for main_index in main_indices))
        elif len(detail_indices) >= 2:
            for position, left in enumerate(detail_indices):
                for right in detail_indices[position + 1 :]:
                    values.append(float(similarity_matrix[left, right].item()))

        return float(median(values)) if values else None

    def _color_consistency_score(
        self, colors: list[tuple[int, int, int]], main_indices: list[int]
    ) -> float:
        indices = main_indices if len(main_indices) >= 2 else list(range(len(colors)))
        if len(indices) < 2:
            return 1.0

        similarities = [
            ColorService.color_similarity(colors[left], colors[right])
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
        ]
        return float(median(similarities)) if similarities else 1.0

    def _weighted_score(self, main_score: float, color_score: float, detail_score: float | None) -> float:
        weighted_values = [
            (main_score, self.settings.main_view_weight),
            (color_score, self.settings.color_consistency_weight),
        ]
        if detail_score is not None:
            weighted_values.append((detail_score, self.settings.detail_support_weight))

        total_weight = sum(weight for _, weight in weighted_values)
        if total_weight == 0:
            return main_score
        return sum(value * weight for value, weight in weighted_values) / total_weight

    def _average_similarity_details(
        self, similarity_matrix: torch.Tensor, view_types: list[ViewTypeResult], main_indices: list[int]
    ) -> dict[int, dict[str, float | None]]:
        image_count = similarity_matrix.shape[0]
        details: dict[int, dict[str, float | None]] = {}

        for estimate in view_types:
            index = estimate.image_index
            same_view_indices = [
                item.image_index for item in view_types if item.view_type == estimate.view_type and item.image_index != index
            ]
            comparable_main_indices = [main_index for main_index in main_indices if main_index != index]

            details[index] = {
                "all": self._average_against(similarity_matrix, index, [item for item in range(image_count) if item != index]),
                "same_view": self._average_against(similarity_matrix, index, same_view_indices),
                "main": self._average_against(similarity_matrix, index, comparable_main_indices),
            }

        return details

    @staticmethod
    def _average_against(similarity_matrix: torch.Tensor, index: int, other_indices: list[int]) -> float | None:
        if not other_indices:
            return None
        return float(torch.tensor([similarity_matrix[index, other].item() for other in other_indices]).mean().item())

    def _flag_images(
        self,
        view_types: list[ViewTypeResult],
        averages: dict[int, dict[str, float | None]],
        dominant_color_rgbs: list[tuple[int, int, int]],
        main_indices: list[int],
    ) -> list[FlaggedImage]:
        flagged_images: list[FlaggedImage] = []

        for estimate in view_types:
            index = estimate.image_index
            view_type = estimate.view_type
            image_averages = averages[index]
            color_similarity = self._average_color_similarity(index, dominant_color_rgbs, main_indices)
            avg_main = image_averages["main"]

            if view_type in MAIN_VIEW_TYPES and avg_main is not None:
                if avg_main < self.settings.main_view_severe_similarity_threshold:
                    flagged_images.append(
                        self._flag(
                            estimate,
                            "high",
                            "Main or partial product view is very different from other main views.",
                            "Review manually before accepting this listing.",
                            image_averages,
                        )
                    )
                    continue
                if avg_main < self.settings.main_view_low_similarity_threshold:
                    flagged_images.append(
                        self._flag(
                            estimate,
                            "medium",
                            "Main or partial product view has low similarity against other main views.",
                            "Review angle, crop, and product identity before rejecting.",
                            image_averages,
                        )
                    )
                    continue

            if color_similarity is not None and color_similarity < self.settings.strong_color_mismatch_threshold:
                severity = "high" if view_type in MAIN_VIEW_TYPES else "medium"
                flagged_images.append(
                    self._flag(
                        estimate,
                        severity,
                        "Dominant color is strongly different from the comparable product images.",
                        "Review manually; color mismatch can indicate a different product or strong lighting issue.",
                        image_averages,
                    )
                )
                continue

            if view_type in DETAIL_VIEW_TYPES and avg_main is not None and avg_main < self.settings.detail_visual_support_threshold:
                flagged_images.append(
                    self._flag(
                        estimate,
                        "low",
                        "Detail image has low global similarity but is not treated as a primary product mismatch.",
                        "Review manually, but do not reject automatically.",
                        image_averages,
                    )
                )

        return flagged_images

    def _average_color_similarity(
        self, index: int, colors: list[tuple[int, int, int]], main_indices: list[int]
    ) -> float | None:
        comparable_indices = [main_index for main_index in main_indices if main_index != index]
        if not comparable_indices:
            comparable_indices = [other_index for other_index in range(len(colors)) if other_index != index]
        if not comparable_indices:
            return None
        return sum(ColorService.color_similarity(colors[index], colors[other]) for other in comparable_indices) / len(
            comparable_indices
        )

    @staticmethod
    def _flag(
        estimate: ViewTypeResult,
        severity: str,
        reason: str,
        recommended_action: str,
        averages: dict[str, float | None],
    ) -> FlaggedImage:
        return FlaggedImage(
            image_index=estimate.image_index,
            view_type=estimate.view_type,
            severity=severity,
            reason=reason,
            recommended_action=recommended_action,
            average_similarity_against_all=round(averages["all"], 4) if averages["all"] is not None else None,
            average_similarity_against_same_view_type=round(averages["same_view"], 4)
            if averages["same_view"] is not None
            else None,
            average_similarity_against_main_views=round(averages["main"], 4) if averages["main"] is not None else None,
        )

    def _status(
        self,
        robust_score: float,
        flagged_images: list[FlaggedImage],
        main_indices: list[int],
        dominant_color_rgbs: list[tuple[int, int, int]],
    ) -> str:
        has_high_main_outlier = any(
            item.severity == "high" and item.view_type in MAIN_VIEW_TYPES for item in flagged_images
        )
        has_strong_main_color_mismatch = self._has_strong_main_color_mismatch(dominant_color_rgbs, main_indices)

        if robust_score < self.settings.review_threshold or has_high_main_outlier or has_strong_main_color_mismatch:
            return "inconsistent"
        if robust_score >= self.settings.consistent_threshold and not flagged_images:
            return "consistent"
        return "needs_review"

    def _has_strong_main_color_mismatch(self, colors: list[tuple[int, int, int]], main_indices: list[int]) -> bool:
        if len(main_indices) < 2:
            return False
        return any(
            ColorService.color_similarity(colors[left], colors[right]) < self.settings.strong_color_mismatch_threshold
            for position, left in enumerate(main_indices)
            for right in main_indices[position + 1 :]
        )
