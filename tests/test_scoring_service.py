import pytest
import torch

from app.config import Settings
from app.schemas.response_schemas import FlaggedImage
from app.services.scoring_service import ScoringService
from app.services.view_type_service import ViewTypeResult


def test_robust_score_with_detail_image_does_not_destroy_listing_score() -> None:
    similarity_matrix = torch.tensor(
        [
            [1.00, 0.90, 0.20, 0.88],
            [0.90, 1.00, 0.22, 0.87],
            [0.20, 0.22, 1.00, 0.18],
            [0.88, 0.87, 0.18, 1.00],
        ]
    )
    view_types = [
        ViewTypeResult(0, "main_front", 0.8),
        ViewTypeResult(1, "main_back", 0.8),
        ViewTypeResult(2, "detail_label", 0.8),
        ViewTypeResult(3, "partial_view", 0.8),
    ]
    colors = [(32, 32, 34), (34, 34, 36), (35, 35, 37), (31, 31, 33)]

    result = ScoringService(Settings()).score(similarity_matrix, view_types, colors)

    assert result.scores.raw_consistency_score == pytest.approx(0.5417, abs=0.0001)
    assert result.scores.main_view_score == pytest.approx(0.88)
    assert result.scores.robust_consistency_score > 0.70


def test_detail_low_global_similarity_is_not_high_severity_by_itself() -> None:
    similarity_matrix = torch.tensor(
        [
            [1.00, 0.90, 0.20, 0.88],
            [0.90, 1.00, 0.22, 0.87],
            [0.20, 0.22, 1.00, 0.18],
            [0.88, 0.87, 0.18, 1.00],
        ]
    )
    view_types = [
        ViewTypeResult(0, "main_front", 0.8),
        ViewTypeResult(1, "main_back", 0.8),
        ViewTypeResult(2, "detail_label", 0.8),
        ViewTypeResult(3, "partial_view", 0.8),
    ]
    colors = [(32, 32, 34), (34, 34, 36), (35, 35, 37), (31, 31, 33)]

    result = ScoringService(Settings()).score(similarity_matrix, view_types, colors)
    detail_flags = [item for item in result.flagged_images if item.image_index == 2]

    assert all(item.severity != "high" for item in detail_flags)
    assert result.status != "inconsistent"


def test_closeup_outlier_does_not_force_inconsistent_when_main_and_color_are_good() -> None:
    similarity_matrix = torch.tensor(
        [
            [1.00, 0.88, 0.86, 0.10],
            [0.88, 1.00, 0.87, 0.12],
            [0.86, 0.87, 1.00, 0.11],
            [0.10, 0.12, 0.11, 1.00],
        ]
    )
    view_types = [
        ViewTypeResult(0, "main_front", 0.8),
        ViewTypeResult(1, "main_back", 0.8),
        ViewTypeResult(2, "partial_view", 0.8),
        ViewTypeResult(3, "detail_label", 0.8),
    ]
    colors = [(30, 30, 30), (31, 31, 31), (29, 29, 29), (32, 32, 32)]

    result = ScoringService(Settings()).score(similarity_matrix, view_types, colors)

    assert result.scores.main_view_score > 0.80
    assert result.scores.color_consistency_score > 0.90
    assert result.status == "consistent"


def test_single_high_main_outlier_with_good_robust_score_needs_review_not_inconsistent() -> None:
    service = ScoringService(Settings())
    view_types = [
        ViewTypeResult(0, "main_front", 0.8),
        ViewTypeResult(1, "main_back", 0.8),
        ViewTypeResult(2, "partial_view", 0.8),
    ]
    flags = [
        FlaggedImage(
            image_index=2,
            view_type="partial_view",
            severity="high",
            issue_type="visual_outlier",
            reason="test",
            recommended_action="review",
        )
    ]

    status = service.resolve_status(0.75, flags, view_types, [(10, 10, 10), (11, 11, 11), (12, 12, 12)])

    assert status == "needs_review"


def test_multiple_high_main_outliers_can_force_inconsistent() -> None:
    service = ScoringService(Settings())
    view_types = [
        ViewTypeResult(0, "main_front", 0.8),
        ViewTypeResult(1, "main_back", 0.8),
        ViewTypeResult(2, "partial_view", 0.8),
    ]
    flags = [
        FlaggedImage(
            image_index=1,
            view_type="main_back",
            severity="high",
            issue_type="visual_outlier",
            reason="test",
            recommended_action="review",
        ),
        FlaggedImage(
            image_index=2,
            view_type="partial_view",
            severity="high",
            issue_type="visual_outlier",
            reason="test",
            recommended_action="review",
        ),
    ]

    status = service.resolve_status(0.75, flags, view_types, [(10, 10, 10), (11, 11, 11), (12, 12, 12)])

    assert status == "inconsistent"
