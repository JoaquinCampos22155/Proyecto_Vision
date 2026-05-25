from scripts.evaluation_utils import (
    acceptable_match,
    exact_match,
    extract_predicted_flagged_filenames,
    flagged_exact_match,
    flagged_partial_match,
)


def test_status_matching_helpers():
    assert exact_match("consistent", "consistent") is True
    assert exact_match("consistent", "needs_review") is False
    assert acceptable_match("consistent", ["consistent", "needs_review"], "needs_review") is True
    assert acceptable_match("consistent", None, "needs_review") is False


def test_flagged_matching_helpers():
    assert flagged_exact_match(["a.jpg"], ["a.jpg"]) is True
    assert flagged_exact_match(["a.jpg"], ["b.jpg"]) is False
    assert flagged_partial_match(["a.jpg", "b.jpg"], ["b.jpg"]) is True
    assert flagged_partial_match([], []) is True
    assert flagged_partial_match(["a.jpg"], []) is False


def test_extract_predicted_flagged_filenames_from_image_index():
    response = {"flagged_images": [{"image_index": 1}]}

    filenames = extract_predicted_flagged_filenames(response, ["a.jpg", "b.jpg"])

    assert filenames == ["b.jpg"]


def test_extract_predicted_flagged_filenames_from_direct_filename():
    response = {"flagged_images": [{"filename": "b.jpg"}, {"original_filename": "c.jpg"}]}

    filenames = extract_predicted_flagged_filenames(response, ["a.jpg", "b.jpg", "c.jpg"])

    assert filenames == ["b.jpg", "c.jpg"]
