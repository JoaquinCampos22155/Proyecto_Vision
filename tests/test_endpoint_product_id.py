from fastapi.testclient import TestClient

from app import main as main_module


def test_product_id_is_received_from_form_data(monkeypatch) -> None:
    async def fake_validate(self, product_id, uploads):
        return {
            "product_id": product_id,
            "image_count": len(uploads),
            "status": "needs_review",
            "scores": {
                "raw_consistency_score": 0.0,
                "main_view_score": None,
                "detail_support_score": None,
                "color_consistency_score": 1.0,
                "robust_consistency_score": 0.0,
                "low_confidence_reason": "test",
            },
            "note": "test",
            "thresholds": {
                "consistent": 0.70,
                "needs_review": 0.50,
                "main_view_low_similarity": 0.45,
                "main_view_severe_similarity": 0.35,
                "strong_color_mismatch": 0.55,
            },
            "view_types": [],
            "flagged_images": [],
            "dominant_colors": [],
            "garment_type_estimates": [],
            "image_debug": [],
            "crop_debug_urls": [],
            "pairwise_similarity_matrix": [],
            "condition_estimate": {
                "label": "unknown",
                "confidence": 0.0,
                "note": "test",
            },
        }

    monkeypatch.setattr(main_module.ProductImageValidationService, "validate", fake_validate)
    client = TestClient(main_module.app)
    files = [("images", (f"image_{index}.jpg", b"fake", "image/jpeg")) for index in range(6)]

    response = client.post("/validate-product-images", data={"product_id": "sku-123"}, files=files)

    assert response.status_code == 200
    assert response.json()["product_id"] == "sku-123"
