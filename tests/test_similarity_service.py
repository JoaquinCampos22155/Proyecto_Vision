import pytest
import torch

from app.services.similarity_service import SimilarityService


def test_similarity_matrix_and_consistency_score_exclude_diagonal() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ]
    )

    matrix = SimilarityService.cosine_similarity_matrix(embeddings)
    score = SimilarityService.consistency_score(matrix)
    robust_score = SimilarityService.robust_consistency_score(matrix)
    averages = SimilarityService.average_similarity_per_image(matrix)

    assert matrix.shape == (3, 3)
    assert torch.allclose(torch.diag(matrix), torch.ones(3))
    assert torch.allclose(matrix, matrix.T)
    assert score == pytest.approx((0.8 + 0.0 + 0.6 + 0.8 + 0.0 + 0.6) / 6)
    assert robust_score == pytest.approx(0.6)
    assert averages == pytest.approx([0.4, 0.7, 0.3])
