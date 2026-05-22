# tests/test_dataset_runner.py

import json
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.services.validation_service import (
    ProductImageValidationService,
)


DATASET_ROOT = Path("tests/data/datasetimagenes")


def get_all_cases():

    cases = []

    for category_dir in DATASET_ROOT.iterdir():

        if not category_dir.is_dir():
            continue

        for version_dir in category_dir.iterdir():

            if not version_dir.is_dir():
                continue

            label_json = version_dir / "label.json"

            if label_json.exists():
                cases.append(version_dir)

    return cases


@pytest.mark.parametrize(
    "case_path",
    get_all_cases()
)
def test_dataset_case(case_path: Path):

    print("\n")
    print("=" * 100)
    print(f"RUNNING CASE: {case_path}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # LOAD LABEL JSON
    # ------------------------------------------------------------------

    label_path = case_path / "label.json"

    with open(label_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    expected_status = metadata["expected_status"]

    expected_flagged_images = sorted(
        metadata["expected_flagged_images"]
    )

    print(f"\nEXPECTED STATUS: {expected_status}")
    print(f"EXPECTED FLAGS: {expected_flagged_images}")

    # ------------------------------------------------------------------
    # LOAD IMAGE PATHS
    # ------------------------------------------------------------------

    image_paths = []

    for image_metadata in metadata["images"]:

        filename = image_metadata["filename"]

        image_path = case_path / filename

        if not image_path.exists():

            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        image_paths.append(image_path)

    print(f"\nLoaded {len(image_paths)} images")

    # ------------------------------------------------------------------
    # INIT VALIDATION SERVICE
    # ------------------------------------------------------------------

    settings = Settings()

    validation_service = ProductImageValidationService(
        settings
    )

    # ------------------------------------------------------------------
    # LOAD REAL IMAGES
    # ------------------------------------------------------------------

    loaded_images = []

    for index, image_path in enumerate(image_paths):

        image = Image.open(image_path).convert("RGB")

        loaded_images.append({
            "index": index,
            "filename": image_path.name,
            "image": image,
        })

    # ------------------------------------------------------------------
    # PREPARE CROPS + VIEW TYPES + COLORS
    # ------------------------------------------------------------------

    cropped_images = []

    dominant_color_rgbs = []

    view_type_results = []

    for loaded in loaded_images:

        detection, cropped, crop_strategy = (
            validation_service._prepare_embedding_image(
                loaded["image"]
            )
        )

        cropped_images.append(cropped)

        view_type_results.append(
            validation_service.view_type_service.estimate(
                cropped,
                image_index=loaded["index"],
                filename=loaded["filename"],
            )
        )

        color_rgb = (
            validation_service.color_service
            .dominant_color_rgb(cropped)
        )

        dominant_color_rgbs.append(color_rgb)

    # ------------------------------------------------------------------
    # EMBEDDINGS + SIMILARITY
    # ------------------------------------------------------------------

    embeddings = (
        validation_service.embedding_service
        .extract_embeddings(cropped_images)
    )

    similarity_matrix = (
        validation_service.similarity_service
        .cosine_similarity_matrix(embeddings)
    )

    # ------------------------------------------------------------------
    # FINAL SCORING
    # ------------------------------------------------------------------

    result = validation_service.scoring_service.score(
        similarity_matrix=similarity_matrix,
        view_types=view_type_results,
        dominant_color_rgbs=dominant_color_rgbs,
    )

    # ------------------------------------------------------------------
    # EXTRACT RESULTS
    # ------------------------------------------------------------------

    predicted_status = result.status

    predicted_flagged_images = sorted([
        flagged.image_index
        for flagged in result.flagged_images
    ])

    # ------------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------------

    print(f"\nPREDICTED STATUS: {predicted_status}")
    print(f"PREDICTED FLAGS: {predicted_flagged_images}")

    print("\nSCORES:")

    print(
        f"raw_consistency_score = "
        f"{result.scores.raw_consistency_score}"
    )

    print(
        f"main_view_score = "
        f"{result.scores.main_view_score}"
    )

    print(
        f"detail_support_score = "
        f"{result.scores.detail_support_score}"
    )

    print(
        f"color_consistency_score = "
        f"{result.scores.color_consistency_score}"
    )

    print(
        f"robust_consistency_score = "
        f"{result.scores.robust_consistency_score}"
    )

    print("\nVIEW TYPES:")

    for item in view_type_results:

        print(
            f"image_{item.image_index} "
            f"-> {item.view_type} "
            f"(confidence={round(item.confidence, 3)})"
        )

    print("\nFLAGGED IMAGES:")

    if not result.flagged_images:
        print("No images flagged")

    for flagged in result.flagged_images:

        print(
            f"\nimage_{flagged.image_index}"
        )

        print(
            f"severity={flagged.severity}"
        )

        print(
            f"reason={flagged.reason}"
        )

        print(
            f"avg_main="
            f"{flagged.average_similarity_against_main_views}"
        )

    print("\nSIMILARITY MATRIX:")

    for row in similarity_matrix.tolist():

        print(
            [round(value, 3) for value in row]
        )

    # ------------------------------------------------------------------
    # SOFT EVALUATION
    # ------------------------------------------------------------------

    status_match = (
        predicted_status == expected_status
    )

    flagged_match = (
        predicted_flagged_images ==
        expected_flagged_images
    )

    print("\nRESULTS:")

    print(
        f"STATUS MATCH: {status_match}"
    )

    print(
        f"FLAGGED MATCH: {flagged_match}"
    )

    # ------------------------------------------------------------------
    # IMPORTANT
    # ------------------------------------------------------------------

    # No hard assertions for now.
    # This phase is for calibration and analysis.

    print("\nCASE COMPLETED")
    print("=" * 100)