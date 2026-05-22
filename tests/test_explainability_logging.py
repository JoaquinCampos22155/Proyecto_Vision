import torch

from app.config import Settings
from app.services.scoring_service import ScoringService
from app.services.view_type_service import ViewTypeResult


def test_explainability_logging_runs_without_breaking_core():

    settings = Settings()

    scoring_service = ScoringService(settings)

    similarity_matrix = torch.tensor([
        [1.00, 0.88, 0.32],
        [0.88, 1.00, 0.29],
        [0.32, 0.29, 1.00],
    ])

    view_types = [
        ViewTypeResult(
            image_index=0,
            view_type="main_front",
            confidence=0.95,
        ),
        ViewTypeResult(
            image_index=1,
            view_type="main_back",
            confidence=0.92,
        ),
        ViewTypeResult(
            image_index=2,
            view_type="detail_label",
            confidence=0.81,
        ),
    ]

    dominant_colors = [
        (30, 120, 40),   # green
        (32, 122, 42),   # green similar
        (10, 10, 10),    # black mismatch
    ]

    result = scoring_service.score(
        similarity_matrix=similarity_matrix,
        view_types=view_types,
        dominant_color_rgbs=dominant_colors,
    )

    # Core system should still work normally
    assert result is not None

    assert result.status in [
        "consistent",
        "needs_review",
        "inconsistent",
    ]

    assert result.scores.robust_consistency_score is not None

    # Flagged images should still exist
    assert isinstance(result.flagged_images, list)

    # Print final result for manual debugging
    print("\n========== FINAL RESULT ==========")
    print(result)

    print("\n========== TEST COMPLETED ==========")