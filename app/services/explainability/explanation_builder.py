from app.services.explainability.explanation_types import (
    LOW_SIMILARITY,
    COLOR_MISMATCH,
    DETAIL_IMAGE,
    OUTLIER_DETECTED,
    LOW_CONFIDENCE,
    HIGH_CONSISTENCY,
    MAIN_VIEW_MATCH,
    POSSIBLE_DIFFERENT_PRODUCT
)


class ExplanationBuilder:

    def __init__(self, logger):
        self.logger = logger

    def evaluate_similarity(
        self,
        image_name: str,
        similarity: float,
        threshold: float
    ):

        if similarity < threshold:

            self.logger.add(
                image=image_name,
                reason_type=LOW_SIMILARITY,
                message=(
                    f"Low similarity detected "
                    f"({similarity:.2f} < {threshold:.2f})"
                ),
                score=similarity
            )

        else:

            self.logger.add(
                image=image_name,
                reason_type=HIGH_CONSISTENCY,
                message=(
                    f"Image is visually consistent "
                    f"({similarity:.2f})"
                ),
                score=similarity
            )

    def evaluate_color(
        self,
        image_name: str,
        color_similarity: float,
        threshold: float
    ):

        if color_similarity < threshold:

            self.logger.add(
                image=image_name,
                reason_type=COLOR_MISMATCH,
                message=(
                    f"Strong color mismatch detected "
                    f"({color_similarity:.2f})"
                ),
                score=color_similarity
            )

    def evaluate_view_type(
        self,
        image_name: str,
        view_type: str
    ):

        if "detail" in view_type:

            self.logger.add(
                image=image_name,
                reason_type=DETAIL_IMAGE,
                message=(
                    f"Image classified as detail/support image "
                    f"({view_type})"
                )
            )

        else:

            self.logger.add(
                image=image_name,
                reason_type=MAIN_VIEW_MATCH,
                message=(
                    f"Image classified as main product view "
                    f"({view_type})"
                )
            )

    def evaluate_outlier(
        self,
        image_name: str,
        similarity_score: float,
        outlier_threshold: float
    ):

        if similarity_score < outlier_threshold:

            self.logger.add(
                image=image_name,
                reason_type=OUTLIER_DETECTED,
                message=(
                    f"Possible outlier detected "
                    f"({similarity_score:.2f})"
                ),
                score=similarity_score
            )

    def evaluate_confidence(
        self,
        image_name: str,
        confidence: float,
        threshold: float
    ):

        if confidence < threshold:

            self.logger.add(
                image=image_name,
                reason_type=LOW_CONFIDENCE,
                message=(
                    f"Low confidence detection "
                    f"({confidence:.2f})"
                ),
                score=confidence
            )

    def evaluate_possible_product_mismatch(
        self,
        image_name: str,
        similarity: float,
        color_similarity: float,
        similarity_threshold: float,
        color_threshold: float
    ):

        if (
            similarity < similarity_threshold and
            color_similarity < color_threshold
        ):

            self.logger.add(
                image=image_name,
                reason_type=POSSIBLE_DIFFERENT_PRODUCT,
                message=(
                    "Image may belong to a different product "
                    "based on similarity and color mismatch"
                ),
                metadata={
                    "similarity": round(similarity, 4),
                    "color_similarity": round(color_similarity, 4)
                }
            )