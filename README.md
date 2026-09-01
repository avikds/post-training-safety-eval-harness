# Post-Training Safety Eval Harness

Build a numpy-only safety eval harness a lab would run after training — not another trainer. The pipeline turns generations and training metadata into a model-card JSON and a below/report/pause release decision, with deterministic metrics for calibration, sycophancy, contamination, fairness, and compute.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** canonicalize_generation_record
- [x] **2.** binary_expected_calibration_error
- [x] **3.** sycophancy_rate
- [x] **4.** exact_match_contamination_rate
- [x] **5.** max_ngram_overlap
- [ ] **6.** demographic_parity_gap
- [ ] **7.** equalized_odds_gap
- [ ] **8.** transformer_training_flops
- [ ] **9.** log10_compute
- [ ] **10.** count_log10_thresholds_met
- [ ] **11.** flagged_eval_names
- [ ] **12.** capability_gate
- [ ] **13.** assemble_model_card
- [ ] **14.** release_decision

---

Built on Deep-ML.
