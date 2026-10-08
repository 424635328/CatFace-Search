"""Diagnose why the metric-learning run did not move the loss.

The observed symptom was a loss pinned at 30.45 (= 64 * ln 352) with no change over ten
epochs, which is the signature of an *inactive* optimisation, not slow convergence. This
script isolates which part of the pipeline is not updating.

Checks performed:
1. Are the backbone's own parameters present in the optimiser groups?
2. Do gradients reach the backbone after one backward pass?
3. Do weights actually change after one optimiser step?
4. Does the loss decrease over a short overfit on a single batch?

Usage::

    python -m tools.diagnose_training
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.logging_utils import configure_utf8_console, get_logger  # noqa: E402
from catface.models.embedder import Embedder, EmbedderConfig  # noqa: E402

LOGGER = get_logger("tools.diagnose_training")


def main() -> int:
    configure_utf8_console()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = EmbedderConfig(
        backbone="dinov2_vits14", embedding_dim=64, pooling="auto",
        head="arcface", head_scale=64.0, head_margin=0.35, image_size=224,
    )
    embedder = Embedder(config, num_classes=16, device=device)
    embedder.train(True)

    print("=== 1. optimiser parameter coverage ===")
    groups = embedder.named_parameter_groups(lr=1e-3, backbone_lr_scale=0.1, weight_decay=0.05)
    for group in groups:
        print(f"  group {group['name']:<14} params={len(group['params']):>5} "
              f"lr={group['lr']:.2e} wd={group['weight_decay']}")
    grouped = {id(p) for group in groups for p in group["params"]}
    network = embedder.backbone.network
    missing = [(name, tuple(p.shape)) for name, p in network.named_parameters() if id(p) not in grouped]
    print(f"  backbone params total={sum(1 for _ in network.parameters())}, "
          f"NOT in optimiser={len(missing)}")
    if missing:
        print(f"  examples: {missing[:5]}")

    print("\n=== 2. gradient flow after one backward ===")
    optimizer = torch.optim.AdamW(groups)
    images = torch.randn(8, 3, 224, 224, device=device)
    labels = torch.randint(0, 16, (8,), device=device)
    features = embedder.pool_features(images)
    print(f"  pooled feature shape: {tuple(features.shape)} (expected 768 for ViT-S: cls+gap)")

    logits = embedder.head(features, labels)
    print(f"  logits shape {tuple(logits.shape)}, "
          f"row0 range [{logits[0].min().item():.3f}, {logits[0].max().item():.3f}], "
          f"std across classes {logits[0].std().item():.4f}")
    loss = torch.nn.functional.cross_entropy(logits, labels)
    print(f"  loss {loss.item():.4f}  (ln(352)*64 = {np.log(352) * 64:.2f}; "
          f"ln(16)*64 = {np.log(16) * 64:.2f})")
    loss.backward()

    with_grad = [name for name, p in network.named_parameters() if p.grad is not None]
    nonzero = [name for name, p in network.named_parameters()
               if p.grad is not None and float(p.grad.abs().sum()) > 0]
    print(f"  backbone params with .grad: {len(with_grad)}/{sum(1 for _ in network.parameters())}")
    print(f"  backbone params with NON-ZERO grad: {len(nonzero)}")
    head_grads = {name: float(p.grad.abs().sum()) for name, p in embedder.head.module.named_parameters()
                  if p.grad is not None}
    print(f"  head gradient magnitudes: {head_grads}")

    print("\n=== 3. do weights change after a step? ===")
    before = {name: p.detach().clone() for name, p in network.named_parameters()}
    head_before = {name: p.detach().clone() for name, p in embedder.head.module.named_parameters()}
    optimizer.step()
    changed = [name for name, p in network.named_parameters()
               if not torch.allclose(before[name], p.detach())]
    head_changed = [name for name, p in embedder.head.module.named_parameters()
                    if not torch.allclose(head_before[name], p.detach())]
    print(f"  backbone params changed: {len(changed)}/{len(before)}")
    if changed:
        print(f"    examples: {changed[:3]}")
    print(f"  head params changed: {len(head_changed)}/{len(head_before)}")

    print("\n=== 4. can it overfit one batch? ===")
    # The decisive test: a working optimisation must drive a single batch's loss toward 0.
    optimizer = torch.optim.AdamW(
        embedder.named_parameter_groups(lr=1e-3, backbone_lr_scale=0.1, weight_decay=0.0)
    )
    for step in range(60):
        optimizer.zero_grad(set_to_none=True)
        features = embedder.pool_features(images)
        loss = torch.nn.functional.cross_entropy(embedder.head(features, labels), labels)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 59:
            print(f"  step {step:3d}: loss {loss.item():.4f}")
    print(f"  final logit std across classes: "
          f"{embedder.head(embedder.pool_features(images), labels)[0].std().item():.4f}")
    print("\nInterpretation: if section 4 does not drive the loss down, the head is not "
          "receiving usable gradients; if it does, the earlier plateau was a "
          "schedule/learning-rate issue rather than a plumbing failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
