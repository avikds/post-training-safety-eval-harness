# post-training-safety-eval-harness
Build a numpy-only safety eval harness a lab would run after training — not another trainer. The pipeline turns generations and training metadata into a model-card JSON and a below/report/pause release decision, with deterministic metrics for calibration, sycophancy, contamination, fairness, and compute.
