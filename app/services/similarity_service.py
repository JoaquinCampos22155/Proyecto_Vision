import torch


class SimilarityService:
    @staticmethod
    def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D tensor shaped as [image_count, embedding_dim].")
        if embeddings.shape[0] < 2:
            raise ValueError("At least two embeddings are required to calculate pairwise similarity.")

        normalized = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
        return normalized @ normalized.T

    @staticmethod
    def consistency_score(similarity_matrix: torch.Tensor) -> float:
        image_count = similarity_matrix.shape[0]
        mask = ~torch.eye(image_count, dtype=torch.bool, device=similarity_matrix.device)
        return float(similarity_matrix[mask].mean().item())

    @staticmethod
    def unique_pairwise_similarities(similarity_matrix: torch.Tensor) -> torch.Tensor:
        image_count = similarity_matrix.shape[0]
        row_indices, column_indices = torch.triu_indices(image_count, image_count, offset=1)
        return similarity_matrix[row_indices, column_indices]

    @staticmethod
    def robust_consistency_score(similarity_matrix: torch.Tensor) -> float:
        pairwise_similarities = SimilarityService.unique_pairwise_similarities(similarity_matrix)
        return float(torch.median(pairwise_similarities).item())

    @staticmethod
    def average_similarity_per_image(similarity_matrix: torch.Tensor) -> list[float]:
        image_count = similarity_matrix.shape[0]
        totals = similarity_matrix.sum(dim=1) - torch.diag(similarity_matrix)
        return (totals / (image_count - 1)).tolist()

    @staticmethod
    def rounded_matrix(similarity_matrix: torch.Tensor, decimals: int = 4) -> list[list[float]]:
        return [[round(float(value), decimals) for value in row] for row in similarity_matrix.tolist()]
