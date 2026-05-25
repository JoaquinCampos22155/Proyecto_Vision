# Evaluation Protocol

The dataset is split into separate evaluation roles:

- `validation`: debugging and regression checks. These cases are known and may be used to understand failures, but new rules should not be tuned only to this set.
- `test`: final measurement on unseen data. Test results should not be used to hand-tune thresholds or case-specific rules.
- `real_world`: optional production-like samples for manual QA and monitoring. These cases may be messy, incomplete, or exploratory.

Every meaningful pipeline change should be compared against the frozen validation baseline in `docs/evaluation_baselines/baseline_v0_3_post_fix.md`.

Recommended commands:

```powershell
python scripts/evaluate_dataset.py --dataset-dir images --output-dir reports --split validation --mode direct --run-name validation_candidate
python scripts/evaluate_dataset.py --dataset-dir images --output-dir reports --split test --mode direct --run-name test_candidate
python scripts/evaluate_dataset.py --dataset-dir images --output-dir reports --split real_world --mode direct --run-name real_world_candidate
```

Interpretation:

- Validation is allowed to guide debugging and regression protection.
- Test is reserved for final measurement after a candidate change is already chosen.
- Real-world data is for qualitative review, drift discovery, and production-readiness checks.
- If validation improves but test drops, treat it as a possible overfitting warning.
- Filename metadata may be used only as weak view-type metadata or for evaluation/debug reporting, never as ground-truth product mismatch evidence.
