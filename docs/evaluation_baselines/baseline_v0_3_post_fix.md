# baseline_v0_3_post_fix

This baseline freezes the post-fix validation behavior for the ecommerce image consistency evaluator.

This is a validation baseline, not proof of production readiness. The current validation set has only 31 cases, and future improvements should not be tuned only to these same cases. Any new scoring, quality, or mismatch rule must be compared against this baseline and then measured on unseen test or real-world data.

## Metrics

Source run: `reports/runs/post_fix_validation`

| Metric | Value |
| --- | ---: |
| total_cases_successfully_evaluated | 31 |
| runtime_error_count | 0 |
| exact_match_accuracy | 0.774194 |
| acceptable_match_accuracy | 1.0 |
| flagged_exact_match_accuracy | 0.83871 |
| flagged_partial_match_accuracy | 0.870968 |

## By Case Type

| case_type | count | exact | acceptable | flagged exact | flagged partial |
| --- | ---: | ---: | ---: | ---: | ---: |
| same_product_normal | 8 | 1.0 | 1.0 | 1.0 | 1.0 |
| same_product_closeups | 6 | 1.0 | 1.0 | 1.0 | 1.0 |
| same_product_hard_conditions | 4 | 0.75 | 1.0 | 0.75 | 0.75 |
| invalid_images | 3 | 0.333333 | 1.0 | 1.0 | 1.0 |
| one_wrong_image | 5 | 0.8 | 1.0 | 0.8 | 1.0 |
| same_color_different_model | 3 | 0.0 | 1.0 | 0.0 | 0.0 |
| same_garment_type_different_color | 2 | 1.0 | 1.0 | 1.0 | 1.0 |

## Current Thresholds And Config Values

These values came from `app/config.py` when the baseline was frozen.

| Setting | Value |
| --- | ---: |
| consistent_threshold | 0.70 |
| review_threshold | 0.50 |
| main_view_weight | 0.65 |
| color_consistency_weight | 0.25 |
| detail_support_weight | 0.05 |
| main_view_low_similarity_threshold | 0.45 |
| main_view_severe_similarity_threshold | 0.35 |
| detail_visual_support_threshold | 0.25 |
| strong_color_mismatch_threshold | 0.55 |
| detail_color_mismatch_threshold | 0.45 |
| detail_outlier_count_for_review | 2 |
| high_main_flags_for_inconsistent | 2 |
| invalid_images_for_inconsistent | 3 |
| color_cluster_similarity_threshold | 0.75 |
| consensus_cluster_low_support_threshold | 0.50 |
| consensus_cluster_gap_threshold | 0.18 |
| use_yolo_crops | false |
| default_crop_strategy | full_image |
| invalid_dark_brightness_threshold | 35.0 |
| invalid_blur_laplacian_threshold | 25.0 |
| invalid_min_resolution_px | 224 |
| invalid_duplicate_hamming_threshold | 1 |
| invalid_duplicate_mean_abs_diff_threshold | 2.0 |

## Expected Behavior By Case Type

- `same_product_normal`: should remain consistent and should not generate product-mismatch flags.
- `same_product_closeups`: detail views should not break otherwise compatible product listings.
- `same_product_hard_conditions`: difficult lighting, wrinkles, shadows, and imperfect crops may produce review, but should not become inconsistent without strong evidence.
- `invalid_images`: invalid or low-quality images should be flagged while preserving valid product evidence.
- `one_wrong_image`: the wrong image should be detected by product-consensus evidence when possible.
- `same_color_different_model`: needs conservative handling; `needs_review` is acceptable unless there is strong evidence of two main-view model clusters.
- `same_garment_type_different_color`: strong color conflicts should be flagged as color mismatch.

## Guardrail

Future changes should preserve protected categories, especially `same_product_normal`, `same_product_closeups`, `same_garment_type_different_color`, `invalid_images`, and `one_wrong_image`. Exact match changes are not automatically failures when acceptable match remains stable, but they must be reported separately.
