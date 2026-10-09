"""Reproduce the stalled run on the real corpus with full instrumentation.

The synthetic loop converges, so the stall depends on the real data or the exact recipe.
This script runs the precise production recipe on real face crops and reports, per step:

* loss, gradient norm before and after clipping, the clipped fraction;
* descriptor space health: mean pairwise cosine between different images in the batch, and
  the per-dimension standard deviation of the embeddings. Both flat-lining at 1.0 and ~0
  respectively is the signature of *embedding collapse*, where every image maps to the same
  vector and the margin loss has no gradient signal to act on.

Usage::

    python -m tools.diagnose_real_training
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.data.manifest import Manifest
from catface.logging_utils import configure_utf8_console, get_logger
from catface.models.embedder import Embedder, EmbedderConfig
from catface.train.loop import IdentityImageDataset, PKBatchSampler

LOGGER = get_logger("tools.diagnose_real")


def main() -> int:
    configure_utf8_console()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_path = Path("data/manifests/cat_individuals_manifest.jsonl")
    splits_dir = Path("data/manifests/cat_individuals_splits")
    ids = {
        line.strip()
        for line in (splits_dir / "train.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    records = [r for r in Manifest.load(manifest_path) if r.image_id in ids]
    identities = sorted({r.identity for r in records})
    print(f"corpus: {len(records)} images / {len(identities)} identities")

    class_to_index = {name: index for index, name in enumerate(identities)}
    dataset = IdentityImageDataset(
        [r.path for r in records],
        [class_to_index[r.identity] for r in records],
        image_size=224,
        train=True,
    )
    sampler = PKBatchSampler(
        [r.identity for r in records],
        identities_per_batch=8,
        samples_per_identity=4,
        batches_per_epoch=max(1, len(records) // 32),
        seed=1337,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True
    )

    embedder = Embedder(
        EmbedderConfig(
            backbone="dinov2_vitb14",
            embedding_dim=512,
            pooling="auto",
            head="arcface",
            head_margin=0.35,
            head_scale=64.0,
            image_size=224,
            gradient_checkpointing=True,
        ),
        num_classes=len(identities),
        device=device,
    )
    embedder.train(True)
    optimizer = torch.optim.AdamW(
        embedder.named_parameter_groups(lr=3e-4, backbone_lr_scale=0.1, weight_decay=0.05)
    )
    precision = "fp32"
    for flag in sys.argv:
        if flag.startswith("--precision="):
            precision = flag.split("=", 1)[1]
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    print("precision:", precision)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    print("\nstep | loss | grad_norm | after_clip | cos_diff_mean | emb_std | logit_std | lr")
    for step, (images, labels) in enumerate(loader, start=1):
        if step > 60:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # The scope must span forward *and* backward; gradient checkpointing recomputes the
        # forward, so it has to see the same positional embedding the graph was built with.
        with embedder.resolution_scope(int(images.shape[-2]), int(images.shape[-1])):
            with torch.amp.autocast("cuda", enabled=precision != "fp32", dtype=dtype):
                features = embedder.pool_features(images)
                logits = embedder.head(features, labels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_norm = 1.0
        for flag in sys.argv:
            if flag.startswith("--clip="):
                clip_norm = float(flag.split("=", 1)[1])
        norm_before = float(torch.nn.utils.clip_grad_norm_(embedder.parameters(), clip_norm))
        norm_after = float(
            sum(p.grad.norm() ** 2 for p in embedder.parameters() if p.grad is not None) ** 0.5
        )
        scaler.step(optimizer)
        scaler.update()

        if step <= 6 or step % 10 == 0:
            with torch.no_grad():
                embeddings = embedder.head.embed(features)
                similarity = embeddings @ embeddings.t()
                off_diagonal = similarity[~torch.eye(len(embeddings), dtype=torch.bool, device=device)]
                lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"{step:4d} | {loss.item():9.3f} | {norm_before:9.3f} | {norm_after:10.3f} | "
                f"{off_diagonal.mean().item():13.4f} | {embeddings.std(dim=0).mean().item():7.4f} | "
                f"{logits.std(dim=1).mean().item():9.4f} | {lr_now:.2e}"
            )

    print("\nReading:")
    print("  cos_diff_mean near 1.0  -> every image maps to nearly the same vector (collapse)")
    print("  emb_std near 0          -> the descriptor has almost no variance across images")
    print("  logit_std near 0        -> the logits are constant, so cross-entropy is stuck at s*ln(C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
