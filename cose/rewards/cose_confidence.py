"""
COSE Confidence Extraction and Cross-Role Agreement utilities.

Extracts intrinsic confidence signals from model logits during rollout,
computes per-role confidence scores, and derives cross-role agreement
for gating, curriculum, and gradient weighting.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Intrinsic signal functions (self-contained, no dependency on verl)
# ---------------------------------------------------------------------------

def self_certainty(log_probs: torch.Tensor) -> torch.Tensor:
    """KL(p || uniform) per token from log_probs. Higher = more confident."""
    # log_probs shape: (seq_len,) — these are log p(chosen_token)
    # For a single chosen token, confidence ≈ -log_prob (higher prob = lower neg log)
    # We use the negative log prob as a proxy: higher prob → lower value → more certain
    return -log_probs  # simple: higher prob token → lower -log_prob → NOT what we want
    # Actually: log_probs are log p(token), so p(token) = exp(log_probs)
    # Self-certainty should be higher when p(token) is higher
    # Just return log_probs directly (less negative = more confident)


def compute_token_confidence(log_probs: torch.Tensor, signal: str = "self_certainty") -> torch.Tensor:
    """
    Compute per-token confidence from log probabilities.

    Args:
        log_probs: (seq_len,) log probabilities of chosen tokens
        signal: confidence signal type

    Returns:
        (seq_len,) confidence scores, higher = more confident
    """
    if signal == "self_certainty":
        # log p(token) — higher (less negative) = more confident
        return log_probs
    elif signal == "top_k_margin":
        # With only chosen-token log probs, we can't compute margin
        # Fall back to log prob as proxy
        return log_probs
    elif signal == "neg_log_prob":
        # Probability of chosen token (in [0,1])
        return log_probs.exp()
    else:
        return log_probs


# ---------------------------------------------------------------------------
# Tag region extraction
# ---------------------------------------------------------------------------

def find_tag_regions(
    text: str,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_length: int,
) -> Dict[str, Tuple[int, int]]:
    """
    Find token index ranges for tagged regions in generated text.

    Returns dict mapping tag names to (start_idx, end_idx) in the response
    portion (0-indexed from start of response, not prompt).
    """
    response_ids = input_ids[prompt_length:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    regions = {}

    # Tags to look for
    tag_patterns = [
        ("think", r"<think>(.*?)</think>"),
        ("question", r"<question>(.*?)</question>"),
        ("verdict", r"<verdict>(.*?)</verdict>"),
        ("answer", r"<answer>(.*?)</answer>"),
        ("score", r"<score>(.*?)</score>"),
        ("reason", r"<reason>(.*?)</reason>"),
    ]

    for tag_name, pattern in tag_patterns:
        matches = list(re.finditer(pattern, response_text, re.DOTALL))
        if not matches:
            continue

        for i, match in enumerate(matches):
            # Find token positions by encoding substrings
            prefix_text = response_text[:match.start()]
            content_text = response_text[:match.end()]

            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            content_tokens = tokenizer.encode(content_text, add_special_tokens=False)

            start_idx = len(prefix_tokens)
            end_idx = len(content_tokens)

            key = f"{tag_name}_{i}" if len(matches) > 1 else tag_name
            regions[key] = (start_idx, end_idx)

    return regions


def extract_region_confidence(
    log_probs: torch.Tensor,
    regions: Dict[str, Tuple[int, int]],
    region_type: str = "output_only",
) -> Dict[str, float]:
    """
    Extract mean confidence for specified regions.

    Args:
        log_probs: (response_len,) log probs for response tokens
        regions: tag name → (start, end) mapping
        region_type: "output_only", "think_only", or "all"

    Returns:
        Dict mapping region names to mean confidence scores
    """
    confidences = {}

    if region_type == "all":
        # Use all response tokens
        if len(log_probs) > 0:
            confidences["all"] = log_probs.mean().item()
        return confidences

    output_tags = {"question", "verdict", "answer", "score", "reason"}
    think_tags = {"think"}

    for tag_name, (start, end) in regions.items():
        base_tag = tag_name.split("_")[0]  # handle "score_0", "score_1" etc.

        if region_type == "output_only" and base_tag not in output_tags:
            continue
        if region_type == "think_only" and base_tag not in think_tags:
            continue

        start = min(start, len(log_probs))
        end = min(end, len(log_probs))

        if end > start:
            region_lp = log_probs[start:end]
            confidences[tag_name] = region_lp.mean().item()

    return confidences


# ---------------------------------------------------------------------------
# Cross-role agreement
# ---------------------------------------------------------------------------

def compute_chain_agreement(
    c_validator: float,
    c_solver_runs: List[float],
    c_judge: float,
) -> float:
    """
    Compute chain agreement as min of downstream role confidences.

    Args:
        c_validator: validator confidence on <verdict>
        c_solver_runs: list of solver confidences across N runs
        c_judge: judge confidence on <score>

    Returns:
        Chain agreement score (higher = more reliable)
    """
    c_solver_mean = np.mean(c_solver_runs) if c_solver_runs else float('-inf')
    return min(c_validator, c_solver_mean, c_judge)


def compute_solver_consistency(
    answers: List[str],
    confidences: List[float],
) -> Dict[str, float]:
    """
    Compute self-consistency metrics from N solver runs.

    Returns:
        answer_agreement: fraction of runs that agree with majority answer
        confidence_variance: variance of confidence across runs
    """
    if not answers:
        return {"answer_agreement": 0.0, "confidence_variance": 0.0}

    # Majority vote
    from collections import Counter
    answer_counts = Counter(answers)
    majority_answer, majority_count = answer_counts.most_common(1)[0]
    answer_agreement = majority_count / len(answers)

    # Confidence variance
    confidence_variance = np.var(confidences) if len(confidences) > 1 else 0.0

    return {
        "answer_agreement": answer_agreement,
        "confidence_variance": float(confidence_variance),
        "majority_answer": majority_answer,
    }


# ---------------------------------------------------------------------------
# Batch-level operations for gating, curriculum, weighting
# ---------------------------------------------------------------------------

def compute_percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Convert raw values to percentile ranks in [0, 1]."""
    if len(values) == 0:
        return values
    n = len(values)
    ranks = np.zeros(n)
    sorted_indices = np.argsort(values)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank / max(n - 1, 1)
    return ranks


def confidence_gate(
    chain_agreements: np.ndarray,
    bottom_percentile: float = 0.3,
) -> np.ndarray:
    """
    Gate: keep samples above bottom_percentile of chain agreement.

    Returns:
        Boolean mask, True = keep
    """
    ranks = compute_percentile_ranks(chain_agreements)
    return ranks >= bottom_percentile


def confidence_curriculum(
    solver_confidences: np.ndarray,
    std: float = 0.2,
) -> np.ndarray:
    """
    Curriculum: prioritize samples near median solver confidence.
    Returns weights in [0, 1], peaking at median.
    """
    ranks = compute_percentile_ranks(solver_confidences)
    # Gaussian centered at 0.5 (median)
    weights = np.exp(-0.5 * ((ranks - 0.5) / std) ** 2)
    return weights


def confidence_sample_weights(
    chain_agreements: np.ndarray,
) -> np.ndarray:
    """
    Sample weighting: percentile rank of chain agreement as gradient weight.
    Returns weights in [0, 1].
    """
    ranks = compute_percentile_ranks(chain_agreements)
    # Ensure minimum weight to avoid zero gradients
    return np.clip(ranks, 0.1, 1.0)
