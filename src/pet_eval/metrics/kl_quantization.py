"""KL-divergence metric comparing fp16 and quantized output distributions.

Used to measure information loss introduced by quantization: lower KL divergence
means the quantized model's output distributions are closer to the original fp16
model's distributions.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as functional

from pet_eval.metrics.types import MetricResult

logger = logging.getLogger(__name__)


def compute_kl_divergence(
    fp16_distributions: list[torch.Tensor],
    quantized_distributions: list[torch.Tensor],
    *,
    threshold: float | None = None,
) -> MetricResult:
    """Compute mean KL divergence between fp16 and quantized output distributions.

    For each (fp16, quantized) distribution pair, clamps both to a minimum of
    1e-10 to avoid log(0), then computes KL(fp16 || quantized) using
    ``torch.nn.functional.kl_div`` with log-probability inputs.

    Args:
        fp16_distributions: Per-sample probability distributions from the fp16 model.
            Each tensor should be a 1-D probability vector summing to ~1.
        quantized_distributions: Corresponding distributions from the quantized model.
            Must have the same length and shape as ``fp16_distributions``.
        threshold: Optional gate threshold (lte operator).  If provided, ``passed``
            will be ``True`` when ``value <= threshold``.  Pass ``None`` for an
            informational-only metric that always passes.

    Returns:
        A single :class:`MetricResult` with:
        - ``name="kl_divergence"``
        - ``value``: mean KL divergence across samples (0.0 for empty inputs)
        - ``details``: ``{"per_sample_kl": [...], "n_samples": N}``
    """
    if not fp16_distributions:
        logger.info("compute_kl_divergence: empty inputs, returning KL=0.0")
        return MetricResult.create(
            "kl_divergence",
            value=0.0,
            threshold=threshold,
            operator="lte",
            details={"per_sample_kl": [], "n_samples": 0},
        )

    per_sample_kl: list[float] = []

    for fp16, quant in zip(fp16_distributions, quantized_distributions):
        # Clamp both distributions to avoid log(0)
        fp16_clamped = fp16.clamp(min=1e-10)
        quant_clamped = quant.clamp(min=1e-10)

        # F.kl_div expects:
        #   input  = log-probabilities (first arg)
        #   target = probabilities     (second arg, log_target=False)
        # KL(fp16 || quant) = sum( fp16 * log(fp16 / quant) )
        # → log_input = log(fp16), target = quant
        kl = functional.kl_div(
            fp16_clamped.log(),
            quant_clamped,
            reduction="sum",
            log_target=False,
        )
        per_sample_kl.append(kl.item())

    mean_kl = sum(per_sample_kl) / len(per_sample_kl)

    logger.info(
        "compute_kl_divergence",
        extra={"n_samples": len(per_sample_kl), "mean_kl": mean_kl},
    )

    return MetricResult.create(
        "kl_divergence",
        value=mean_kl,
        threshold=threshold,
        operator="lte",
        details={"per_sample_kl": per_sample_kl, "n_samples": len(per_sample_kl)},
    )
