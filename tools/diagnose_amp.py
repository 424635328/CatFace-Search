"""Instrument the training step to find why the loss ratio stayed flat.

The plumbing check (``tools/diagnose_training.py``) proved the model, head and optimiser
all work in isolation, so the fault is in the loop itself. This script runs the real loop
step-by-step and counts what the AMP gradient scaler did, because a scaler that repeatedly
skips the optimiser step produces exactly the observed symptom: finite, non-decreasing
loss with a model that quietly never updates.

Usage::

    python -m tools.diagnose_amp
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.logging_utils import configure_utf8_console, get_logger
from catface.models.embedder import Embedder, EmbedderConfig

LOGGER = get_logger("tools.diagnose_amp")


def run_steps(
    embedder: Embedder,
    device: str,
    amp: bool,
    steps: int = 200,
    batch: int = 32,
    classes: int = 32,
    lr: float = 3e-4,
) -> dict:
    """Run ``steps`` real training steps and report update accounting."""
    import torch

    optimizer = torch.optim.AdamW(
        embedder.named_parameter_groups(lr=lr, backbone_lr_scale=0.1, weight_decay=0.05)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
    losses: list[float] = []
    skipped = 0
    scales: list[float] = []
    before = {name: p.detach().clone() for name, p in embedder.backbone.network.named_parameters()}
    grad_norms: list[float] = []

    for _step in range(steps):
        # Fresh random batch each step, like the real loop.
        images = torch.randn(batch, 3, 224, 224, device=device)
        labels = torch.randint(0, classes, (batch,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
            features = embedder.pool_features(images)
            loss = torch.nn.functional.cross_entropy(embedder.head(features, labels), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
        grad_norms.append(float(total_norm))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped += 1
        scales.append(scaler.get_scale())
        losses.append(float(loss.detach()))

    changed = sum(
        1
        for name, p in embedder.backbone.network.named_parameters()
        if not torch.allclose(before[name], p.detach())
    )
    return {
        "amp": amp,
        "steps": steps,
        "scaler_skipped_steps": skipped,
        "final_grad_scale": scales[-1] if scales else None,
        "max_grad_scale": max(scales) if scales else None,
        "backbone_params_changed": changed,
        "total_backbone_params": len(before),
        "loss_first10_mean": round(float(np.mean(losses[:10])), 3),
        "loss_last10_mean": round(float(np.mean(losses[-10:])), 3),
    }


def main() -> int:
    configure_utf8_console()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}\n")

    for amp in (False, True):
        torch.manual_seed(0)
        embedder = Embedder(
            EmbedderConfig(
                backbone="dinov2_vits14", embedding_dim=64, pooling="auto", head="arcface", image_size=224
            ),
            num_classes=32,
            device=device,
        )
        embedder.train(True)
        report = run_steps(embedder, device, amp=amp)
        print(f"=== amp={amp} ===")
        for key, value in report.items():
            if key != "amp":
                print(f"  {key}: {value}")
        print()

    print(
        "A non-zero 'scaler_skipped_steps' means AMP discarded that many updates, which "
        "freezes the model while still reporting a finite loss."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
