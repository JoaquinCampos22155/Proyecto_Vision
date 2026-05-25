from dataclasses import dataclass
from statistics import median

import torch

from app.config import Settings
from app.schemas.response_schemas import FlaggedImage, ScoreBreakdown
from app.services.color_service import ColorService
from app.services.explainability import decision_logger
from app.services.similarity_service import SimilarityService
from app.services.view_type_service import DETAIL_VIEW_TYPES, MAIN_VIEW_TYPES, ViewTypeResult
from app.services.explainability.decision_logger import DecisionLogger
from app.services.explainability.explanation_builder import ExplanationBuilder

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
        decision_logger = DecisionLogger()
        explainer = ExplanationBuilder(decision_logger)
        main_indices = [
            estimate.image_index for estimate in view_types if estimate.view_type in MAIN_VIEW_TYPES
        ]
        detail_indices = [
            estimate.image_index for estimate in view_types if estimate.view_type in DETAIL_VIEW_TYPES
        ]
        for estimate in view_types:
            explainer.evaluate_view_type(
                image_name=f"image_{estimate.image_index}",
                view_type=estimate.view_type,
            )

        main_view_score = self._score_indices(similarity_matrix, main_indices)
        detail_support_score = self._detail_support_score(similarity_matrix, detail_indices, main_indices)
        color_consistency_score = self._color_consistency_score(dominant_color_rgbs, main_indices)
        model_mismatch_detected = self._has_model_mismatch_clusters(
            similarity_matrix=similarity_matrix,
            view_types=view_types,
            colors=dominant_color_rgbs,
        )

        low_confidence_reason = None
        if main_view_score is None:
            low_confidence_reason = "Fewer than two main or partial views were detected; raw score is used as fallback."

        robust_score = self._weighted_score(
            main_score=main_view_score if main_view_score is not None else raw_score,
            color_score=color_consistency_score,
            detail_score=detail_support_score,
        )

        averages = self._average_similarity_details(similarity_matrix, view_types, main_indices)
        for estimate in view_types:

            image_index = estimate.image_index
            avg_main = averages[image_index]["main"]

            if avg_main is not None:

                explainer.evaluate_similarity(
                    image_name=f"image_{image_index}",
                    similarity=avg_main,
                    threshold=self.settings.main_view_low_similarity_threshold,
                )
        for estimate in view_types:

            image_index = estimate.image_index

            color_similarity = self._average_color_similarity(
                image_index,
                dominant_color_rgbs,
                main_indices,
            )

            if color_similarity is not None:

                explainer.evaluate_color(
                    image_name=f"image_{image_index}",
                    color_similarity=color_similarity,
                    threshold=self.settings.strong_color_mismatch_threshold,
                )
        for estimate in view_types:

            image_index = estimate.image_index

            avg_main = averages[image_index]["main"]

            color_similarity = self._average_color_similarity(
                image_index,
                dominant_color_rgbs,
                main_indices,
            )

            if avg_main is not None and color_similarity is not None:

                explainer.evaluate_possible_product_mismatch(
                    image_name=f"image_{image_index}",
                    similarity=avg_main,
                    color_similarity=color_similarity,
                    similarity_threshold=self.settings.main_view_low_similarity_threshold,
                    color_threshold=self.settings.strong_color_mismatch_threshold,
                )
                
        
        flagged_images = self._flag_images(
            similarity_matrix=similarity_matrix,
            view_types=view_types,
            averages=averages,
            dominant_color_rgbs=dominant_color_rgbs,
            main_indices=main_indices,
            main_view_score=main_view_score,
            color_consistency_score=color_consistency_score,
        )
        for flagged in flagged_images:

            similarity_score = (
                flagged.average_similarity_against_main_views
            )

            if similarity_score is not None:

                explainer.evaluate_outlier(
                    image_name=f"image_{flagged.image_index}",
                    similarity_score=similarity_score,
                    outlier_threshold=self.settings.main_view_severe_similarity_threshold,
                )
                
        
        status = self.resolve_status(
            robust_score=robust_score,
            flagged_images=flagged_images,
            view_types=view_types,
            dominant_color_rgbs=dominant_color_rgbs,
            model_mismatch_detected=model_mismatch_detected,
        )
        if status == "consistent" and self._has_product_uncertainty_gap(
            raw_score=raw_score,
            main_view_score=main_view_score,
            detail_support_score=detail_support_score,
            view_types=view_types,
        ):
            status = "needs_review"
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

    @staticmethod
    def _has_product_uncertainty_gap(
        raw_score: float,
        main_view_score: float | None,
        detail_support_score: float | None,
        view_types: list[ViewTypeResult],
    ) -> bool:
        main_count = sum(1 for item in view_types if item.view_type in MAIN_VIEW_TYPES)
        detail_count = sum(1 for item in view_types if item.view_type in DETAIL_VIEW_TYPES)
        return (
            main_count >= 3
            and detail_count >= 2
            and raw_score < 0.69
            and main_view_score is not None
            and main_view_score < 0.80
            and detail_support_score is not None
            and detail_support_score < 0.75
        )

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
        similarity_matrix: torch.Tensor,
        view_types: list[ViewTypeResult],
        averages: dict[int, dict[str, float | None]],
        dominant_color_rgbs: list[tuple[int, int, int]],
        main_indices: list[int],
        main_view_score: float | None,
        color_consistency_score: float,
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
                            "visual_outlier",
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
                            "visual_outlier",
                            "Main or partial product view has low similarity against other main views.",
                            "Review angle, crop, and product identity before rejecting.",
                            image_averages,
                        )
                    )
                    continue

            color_threshold = (
                self.settings.detail_color_mismatch_threshold
                if view_type in DETAIL_VIEW_TYPES
                else self.settings.strong_color_mismatch_threshold
            )
            if color_similarity is not None and color_similarity < color_threshold:
                severity = "high" if view_type in MAIN_VIEW_TYPES else "medium"
                flagged_images.append(
                    self._flag(
                        estimate,
                        severity,
                        "color_mismatch",
                        "Dominant color is strongly different from the comparable product images.",
                        "Review manually; color mismatch can indicate a different product or strong lighting issue.",
                        image_averages,
                    )
                )
                continue

        main_and_color_are_healthy = (
            main_view_score is not None
            and main_view_score >= self.settings.consistent_threshold
            and color_consistency_score >= self.settings.consistent_threshold
        )
        detail_support_threshold = (
            self.settings.detail_visual_support_threshold
            if main_and_color_are_healthy
            else self.settings.main_view_low_similarity_threshold
        )
        weak_detail_candidates = [
            estimate
            for estimate in view_types
            if estimate.view_type in DETAIL_VIEW_TYPES
            and averages[estimate.image_index]["main"] is not None
            and averages[estimate.image_index]["main"] < detail_support_threshold
        ]
        main_views_are_weak = main_view_score is None or main_view_score < self.settings.review_threshold
        if main_views_are_weak and weak_detail_candidates:
            for estimate in weak_detail_candidates:
                flagged_images.append(
                    self._flag(
                        estimate,
                        "low",
                        "detail_low_support",
                        "Detail image has low visual support, but detail views are not treated as primary product mismatches.",
                        "Review manually, but do not reject automatically.",
                        averages[estimate.image_index],
                    )
                )

        flagged_images.extend(
            self._consensus_cluster_flags(
                similarity_matrix=similarity_matrix,
                view_types=view_types,
                existing_flags=flagged_images,
                averages=averages,
            )
        )
        flagged_images.extend(
            self._model_mismatch_cluster_flags(
                similarity_matrix=similarity_matrix,
                view_types=view_types,
                colors=dominant_color_rgbs,
                existing_flags=flagged_images,
                averages=averages,
            )
        )
        flagged_images.extend(
            self._main_color_cluster_flags(
                view_types=view_types,
                colors=dominant_color_rgbs,
                existing_flags=flagged_images,
                averages=averages,
            )
        )
        return flagged_images

    def _consensus_cluster_flags(
        self,
        similarity_matrix: torch.Tensor,
        view_types: list[ViewTypeResult],
        existing_flags: list[FlaggedImage],
        averages: dict[int, dict[str, float | None]],
    ) -> list[FlaggedImage]:
        main_indices = [item.image_index for item in view_types if item.view_type in MAIN_VIEW_TYPES]
        if len(main_indices) < 4:
            return []

        already_flagged = {item.image_index for item in existing_flags}
        support_by_index: dict[int, float] = {}
        for index in main_indices:
            similarities = sorted(
                [float(similarity_matrix[index, other].item()) for other in main_indices if other != index],
                reverse=True,
            )
            top_k = similarities[: min(3, len(similarities))]
            support_by_index[index] = sum(top_k) / len(top_k)

        support_values = list(support_by_index.values())
        median_support = float(median(support_values))
        flags: list[FlaggedImage] = []
        by_index = {item.image_index: item for item in view_types}
        for index, support in support_by_index.items():
            if index in already_flagged:
                continue
            if (
                support < self.settings.consensus_cluster_low_support_threshold
                and support <= median_support - self.settings.consensus_cluster_gap_threshold
            ):
                estimate = by_index[index]
                flags.append(
                    self._flag(
                        estimate,
                        "high",
                        "consensus_outlier",
                        "Main product view has weak support against the strongest consensus cluster.",
                        "Review manually; this image may belong to a different product.",
                        averages[index],
                    )
                )
        return flags

    def _model_mismatch_cluster_flags(
        self,
        similarity_matrix: torch.Tensor,
        view_types: list[ViewTypeResult],
        colors: list[tuple[int, int, int]],
        existing_flags: list[FlaggedImage],
        averages: dict[int, dict[str, float | None]],
    ) -> list[FlaggedImage]:
        clusters = self._model_mismatch_clusters(similarity_matrix, view_types, colors)
        if not clusters:
            return []

        cluster_a, cluster_b = clusters
        if len(cluster_a) == len(cluster_b):
            return []

        minority = cluster_a if len(cluster_a) < len(cluster_b) else cluster_b
        already_flagged = {item.image_index for item in existing_flags}
        by_index = {item.image_index: item for item in view_types}
        flags: list[FlaggedImage] = []
        for index in minority:
            if index in already_flagged:
                continue
            flags.append(
                self._flag(
                    by_index[index],
                    "high",
                    "model_mismatch",
                    "Main-view clustering found a visually distinct product-model cluster with similar color.",
                    "Review manually; this may indicate a different model rather than a detail or lighting difference.",
                    averages[index],
                )
            )
        return flags

    def _has_model_mismatch_clusters(
        self,
        similarity_matrix: torch.Tensor,
        view_types: list[ViewTypeResult],
        colors: list[tuple[int, int, int]],
    ) -> bool:
        return self._model_mismatch_clusters(similarity_matrix, view_types, colors) is not None

    def _model_mismatch_clusters(
        self,
        similarity_matrix: torch.Tensor,
        view_types: list[ViewTypeResult],
        colors: list[tuple[int, int, int]],
    ) -> tuple[list[int], list[int]] | None:
        main_indices = [item.image_index for item in view_types if item.view_type in {"main_front", "main_back"}]
        if len(main_indices) < 4:
            return None

        groups: list[list[int]] = []
        for index in main_indices:
            for group in groups:
                group_similarity = sum(float(similarity_matrix[index, other].item()) for other in group) / len(group)
                if group_similarity >= self.settings.model_cluster_within_similarity_threshold:
                    group.append(index)
                    break
            else:
                groups.append([index])

        candidate_groups = [group for group in groups if len(group) >= 2]
        if len(candidate_groups) != 2:
            return None

        group_a, group_b = candidate_groups
        within_scores = [
            self._score_indices(similarity_matrix, group_a),
            self._score_indices(similarity_matrix, group_b),
        ]
        if any(score is None or score < self.settings.model_cluster_within_similarity_threshold for score in within_scores):
            return None

        between_scores = [
            float(similarity_matrix[left, right].item())
            for left in group_a
            for right in group_b
        ]
        if not between_scores or float(median(between_scores)) > self.settings.model_cluster_between_similarity_threshold:
            return None

        color_scores = [
            ColorService.color_similarity(colors[left], colors[right])
            for left in group_a
            for right in group_b
        ]
        if not color_scores or float(median(color_scores)) < self.settings.model_cluster_color_similarity_threshold:
            return None

        return group_a, group_b

    def _main_color_cluster_flags(
        self,
        view_types: list[ViewTypeResult],
        colors: list[tuple[int, int, int]],
        existing_flags: list[FlaggedImage],
        averages: dict[int, dict[str, float | None]],
    ) -> list[FlaggedImage]:
        main_indices = [item.image_index for item in view_types if item.view_type in MAIN_VIEW_TYPES]
        if len(main_indices) < 3:
            return []

        groups: list[list[int]] = []
        for index in main_indices:
            for group in groups:
                similarity_to_group = sum(
                    ColorService.color_similarity(colors[index], colors[other]) for other in group
                ) / len(group)
                if similarity_to_group >= self.settings.color_cluster_similarity_threshold:
                    group.append(index)
                    break
            else:
                groups.append([index])

        if len(groups) < 2:
            return []

        group_support = [self._color_group_support(group, colors) for group in groups]
        max_support = max(group_support)
        minority_groups = [
            group
            for group, support in zip(groups, group_support)
            if support < max_support
        ]
        if not minority_groups:
            return []

        already_flagged = {item.image_index for item in existing_flags}
        by_index = {item.image_index: item for item in view_types}
        flags: list[FlaggedImage] = []
        for group in minority_groups:
            for index in group:
                if index in already_flagged:
                    continue
                flags.append(
                    self._flag(
                        by_index[index],
                        "high",
                        "color_mismatch",
                        "Main-view color clustering found this image in a minority color group.",
                        "Review manually; this can indicate the same garment type in a different color.",
                        averages[index],
                    )
                )
        return flags

    def _color_group_support(self, group: list[int], colors: list[tuple[int, int, int]]) -> int:
        centroid = tuple(
            int(sum(colors[index][channel] for index in group) / len(group))
            for channel in range(3)
        )
        return sum(
            1
            for color in colors
            if ColorService.color_similarity(color, centroid) >= self.settings.color_cluster_similarity_threshold
        )

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
        issue_type: str,
        reason: str,
        recommended_action: str,
        averages: dict[str, float | None],
    ) -> FlaggedImage:
        return FlaggedImage(
            image_index=estimate.image_index,
            filename=estimate.filename or None,
            original_filename=estimate.filename or None,
            view_type=estimate.view_type,
            severity=severity,
            issue_type=issue_type,
            reason=reason,
            recommended_action=recommended_action,
            average_similarity_against_all=round(averages["all"], 4) if averages["all"] is not None else None,
            average_similarity_against_same_view_type=round(averages["same_view"], 4)
            if averages["same_view"] is not None
            else None,
            average_similarity_against_main_views=round(averages["main"], 4) if averages["main"] is not None else None,
        )

    def resolve_status(
        self,
        robust_score: float,
        flagged_images: list[FlaggedImage],
        view_types: list[ViewTypeResult],
        dominant_color_rgbs: list[tuple[int, int, int]],
        model_mismatch_detected: bool = False,
    ) -> str:
        main_indices = [
            estimate.image_index for estimate in view_types if estimate.view_type in MAIN_VIEW_TYPES
        ]
        high_main_flags = [
            item
            for item in flagged_images
            if item.severity == "high" and item.view_type in MAIN_VIEW_TYPES
        ]
        medium_flags = [item for item in flagged_images if item.severity == "medium"]
        invalid_flags = [
            item for item in flagged_images if item.view_type == "invalid_or_low_quality"
        ]
        severe_invalid_flags = [
            item
            for item in invalid_flags
            if item.severity == "high" or item.issue_type in {"no_garment_visible", "screenshot", "other_object"}
        ]
        has_strong_main_color_mismatch = self._has_strong_main_color_mismatch(dominant_color_rgbs, main_indices)
        has_explicit_product_conflict = any(
            item.issue_type in {"different_product", "color_mismatch", "model_mismatch"}
            and item.severity == "high"
            and item.view_type in MAIN_VIEW_TYPES
            for item in flagged_images
        )

        if robust_score < self.settings.review_threshold:
            return "inconsistent"

        if (
            len(high_main_flags) >= self.settings.high_main_flags_for_inconsistent
            or has_strong_main_color_mismatch
            or model_mismatch_detected
            or len(invalid_flags) >= self.settings.invalid_images_for_inconsistent
            or len(severe_invalid_flags) >= 2
            or has_explicit_product_conflict
        ):
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
