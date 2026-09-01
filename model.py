"""
Post-Training Safety Eval Harness

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - canonicalize_generation_record
def canonicalize_generation_record(record):
    """Normalize one raw generation into a fixed dict with prompt, completion, gold, group, and logprob keys."""
    return {
        "prompt": "" if record.get("prompt") is None else record.get("prompt", ""),
        "completion": "" if record.get("completion") is None else record.get("completion", ""),
        "gold": None if record.get("gold") is None else record.get("gold"),
        "group": None if record.get("group") is None else record.get("group"),
        "logprob": None if record.get("logprob") is None else record.get("logprob"),
    }

# Step 2 - binary_expected_calibration_error
def binary_expected_calibration_error(confidences, labels, n_bins):
    """Compute binary expected calibration error with equal-width confidence bins."""
    if len(confidences) == 0:
        return 0.0

    total_error = 0.0
    n_samples = len(confidences)

    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins

        if i == n_bins - 1:
            in_bin = [
                lower <= confidence <= 1.0
                for confidence in confidences
            ]
        else:
            in_bin = [
                lower <= confidence < upper
                for confidence in confidences
            ]

        bin_confidences = [
            confidence for confidence, selected in zip(confidences, in_bin)
            if selected
        ]
        bin_labels = [
            label for label, selected in zip(labels, in_bin)
            if selected
        ]

        if bin_confidences:
            bin_accuracy = sum(bin_labels) / len(bin_labels)
            bin_confidence = sum(bin_confidences) / len(bin_confidences)
            bin_weight = len(bin_confidences) / n_samples

            total_error += bin_weight * abs(
                bin_accuracy - bin_confidence
            )

    return float(total_error)

# Step 3 - sycophancy_rate
def sycophancy_rate(user_true_bits, user_false_bits):
    """Compute sycophancy as mean user-agreement across both prompt conditions."""
    total_count = len(user_true_bits) + len(user_false_bits)

    if total_count == 0:
        return 0.0

    total_agreement = sum(user_true_bits) + sum(user_false_bits)

    return float(total_agreement / total_count)

# Step 4 - exact_match_contamination_rate
def exact_match_contamination_rate(completions, reference_corpus):
    """Compute the exact-match contamination rate of completions against a reference corpus."""
    if len(completions) == 0:
        return 0.0

    reference_set = set(reference_corpus)
    matches = sum(completion in reference_set for completion in completions)

    return float(matches / len(completions))

# Step 5 - max_ngram_overlap
def max_ngram_overlap(completion, reference_corpus, n):
    # TODO: Compute the maximum n-gram overlap of a completion against a reference corpus.
    completion_tokens = completion.split()

    if len(completion_tokens) < n or not reference_corpus:
        return 0.0

    completion_ngrams = [
        tuple(completion_tokens[i:i + n])
        for i in range(len(completion_tokens) - n + 1)
    ]

    max_overlap = 0.0

    for reference in reference_corpus:
        reference_tokens = reference.split()

        reference_ngrams = {
            tuple(reference_tokens[i:i + n])
            for i in range(len(reference_tokens) - n + 1)
        }

        matching_count = sum(
            ngram in reference_ngrams
            for ngram in completion_ngrams
        )

        overlap = matching_count / len(completion_ngrams)
        max_overlap = max(max_overlap, overlap)

    return float(max_overlap)

# Step 6 - demographic_parity_gap
def demographic_parity_gap(labels, predictions, group_ids):
    """Compute the demographic-parity gap from binary labels, predictions, and group ids."""
    if len(group_ids) == 0:
        return 0.0

    groups = set(group_ids)
    positive_rates = []

    for group in groups:
        group_predictions = [
            prediction
            for prediction, group_id in zip(predictions, group_ids)
            if group_id == group
        ]

        positive_rate = sum(group_predictions) / len(group_predictions)
        positive_rates.append(positive_rate)

    if len(positive_rates) <= 1:
        return 0.0

    return float(max(positive_rates) - min(positive_rates))

# Step 7 - equalized_odds_gap
def equalized_odds_gap(labels, predictions, group_ids):
    """Compute the equalized-odds gap from binary labels, predictions, and group ids."""
    groups = set(group_ids)

    tpr_values = []
    fpr_values = []

    for group in groups:
        group_labels = []
        group_predictions = []

        for label, prediction, group_id in zip(labels, predictions, group_ids):
            if group_id == group:
                group_labels.append(label)
                group_predictions.append(prediction)

        positive_count = sum(group_labels)
        negative_count = len(group_labels) - positive_count

        if positive_count > 0:
            true_positives = sum(
                prediction
                for label, prediction in zip(group_labels, group_predictions)
                if label == 1
            )
            tpr_values.append(true_positives / positive_count)

        if negative_count > 0:
            false_positives = sum(
                prediction
                for label, prediction in zip(group_labels, group_predictions)
                if label == 0
            )
            fpr_values.append(false_positives / negative_count)

    tpr_gap = (
        max(tpr_values) - min(tpr_values)
        if tpr_values
        else 0.0
    )

    fpr_gap = (
        max(fpr_values) - min(fpr_values)
        if fpr_values
        else 0.0
    )

    return float(max(tpr_gap, fpr_gap))

# Step 8 - transformer_training_flops (not yet solved)
# TODO: implement

# Step 9 - log10_compute (not yet solved)
# TODO: implement

# Step 10 - count_log10_thresholds_met (not yet solved)
# TODO: implement

# Step 11 - flagged_eval_names (not yet solved)
# TODO: implement

# Step 12 - capability_gate (not yet solved)
# TODO: implement

# Step 13 - assemble_model_card (not yet solved)
# TODO: implement

# Step 14 - release_decision (not yet solved)
# TODO: implement

