"""Training-loop tests.

The loop is where the expensive mistakes live: a batch that contains no positive pairs
teaches nothing, and a validation metric that is not identity-disjoint reports a model
that memorised the training set. Both are tested here on tiny synthetic data so the suite
stays fast and CPU-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# ruff: noqa: E402 - these imports must follow importorskip, which would otherwise be
# reported as an unused import and the suite would fail rather than skip without torch.

from catface.data.manifest import FaceRecord
from catface.errors import ModelError
from catface.eval.protocols import build_identity_split
from catface.models.embedder import Embedder, EmbedderConfig
from catface.train.loop import (
    IdentityImageDataset,
    PKBatchSampler,
    TrainConfigResolved,
    Trainer,
)


def synthetic_records(tmp_path: Path, identities: int = 6, per_identity: int = 4) -> list[FaceRecord]:
    """Write a tiny on-disk corpus and return records for it."""
    import cv2

    records: list[FaceRecord] = []
    rng = np.random.default_rng(0)
    for index in range(identities):
        for member in range(per_identity):
            # Each identity gets a distinct base colour so a model can separate them.
            colour = np.array([(index * 37) % 256, (index * 91) % 256, (index * 53) % 256], np.uint8)
            image = np.full((64, 64, 3), colour, np.uint8)
            noise = rng.integers(0, 20, image.shape, dtype=np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            path = tmp_path / f"id{index}_{member}.jpg"
            cv2.imwrite(str(path), image)
            records.append(
                FaceRecord(
                    image_id=f"t:id{index}_{member}",
                    path=str(path),
                    source="t",
                    identity=f"id{index}",
                )
            )
    return records


class TestPKBatchSampler:
    def _sampler(self, identities: int = 5, per_identity: int = 4, p: int = 3, k: int = 2) -> PKBatchSampler:
        labels = [f"id{i}" for i in range(identities) for _ in range(per_identity)]
        return PKBatchSampler(
            labels, identities_per_batch=p, samples_per_identity=k, batches_per_epoch=4, seed=1
        )

    def test_batch_has_the_requested_shape(self):
        sampler = self._sampler()
        for batch in sampler:
            assert len(batch) == 6  # p=3 identities x k=2 samples

    def test_identities_with_a_single_image_are_excluded(self):
        labels = ["solo"] + [f"id{i}" for i in range(4) for _ in range(3)]
        sampler = PKBatchSampler(labels, identities_per_batch=2, samples_per_identity=2, batches_per_epoch=2)
        assert "solo" not in sampler.identities

    def test_no_positive_pairs_is_rejected_loudly(self):
        """A corpus of singletons cannot train a metric learner; say so up front."""
        labels = [f"solo{i}" for i in range(10)]
        with pytest.raises(ModelError, match="at least 2 training images"):
            PKBatchSampler(labels, identities_per_batch=2, samples_per_identity=2, batches_per_epoch=2)

    def test_every_batch_contains_positive_pairs(self):
        """The property PK sampling exists for: no batch is positive-pair-free."""
        sampler = self._sampler()
        for batch in sampler:
            labels = [f"id{i}" for i in range(5) for _ in range(4)]
            batch_labels = [labels[i] for i in batch]
            counts = {label: batch_labels.count(label) for label in set(batch_labels)}
            assert any(count >= 2 for count in counts.values())

    def test_epochs_differ_but_are_reproducible(self):
        sampler = self._sampler()
        sampler.set_epoch(0)
        first = [list(batch) for batch in sampler]
        sampler.set_epoch(0)
        assert [list(batch) for batch in sampler] == first
        sampler.set_epoch(1)
        assert [list(batch) for batch in sampler] != first

    def test_samples_within_a_batch_are_distinct_when_possible(self):
        sampler = self._sampler(p=2, k=3)
        for batch in sampler:
            # With k=3 out of 4 available images per identity, no duplicates are needed.
            assert len(set(batch)) == len(batch)


class TestIdentityImageDataset:
    def test_returns_tensor_and_label(self, tmp_path):
        records = synthetic_records(tmp_path, identities=2, per_identity=2)
        dataset = IdentityImageDataset([r.path for r in records], [0, 0, 1, 1], image_size=32)
        image, label = dataset[0]
        assert image.shape == (3, 32, 32)
        assert label in (0, 1)

    def test_eval_transform_is_deterministic(self, tmp_path):
        records = synthetic_records(tmp_path, identities=1, per_identity=2)
        dataset = IdentityImageDataset([r.path for r in records], [0, 0], image_size=32, train=False)
        assert torch.allclose(dataset[0][0], dataset[0][0])

    def test_unreadable_file_yields_a_blank_tile_instead_of_crashing(self, tmp_path):
        """A corrupt image in a 13k-image corpus must not abort an epoch."""
        good = synthetic_records(tmp_path, identities=1, per_identity=1)[0]
        dataset = IdentityImageDataset([good.path, str(tmp_path / "missing.jpg")], [0, 0], image_size=32)
        image, _ = dataset[1]
        assert image.shape == (3, 32, 32)
        assert float(image.abs().sum()) == 0.0
        assert dataset.failures == 1


class TestTrainerEndToEnd:
    def _setup(self, tmp_path, identities: int = 6, per_identity: int = 4):
        records = synthetic_records(tmp_path, identities=identities, per_identity=per_identity)
        config = EmbedderConfig(
            backbone="timm",
            timm_name="resnet10t.c3_in1k",
            embedding_dim=16,
            pooling="gap",
            head="arcface",
            image_size=32,
        )
        embedder = Embedder(config, num_classes=identities, device="cpu")
        val_split = build_identity_split(records, queries_per_identity=1, seed=0, name="val")
        train_config = TrainConfigResolved(
            epochs=2,
            batch_size=4,
            lr=1e-3,
            identities_per_batch=2,
            samples_per_identity=2,
            val_every=1,
            num_workers=0,
            image_size=32,
            output_dir=tmp_path / "train",
            amp=False,
            warmup_epochs=0,
            triplet_weight=0.3,
        )
        return records, embedder, val_split, train_config

    def test_two_epochs_run_and_record_history(self, tmp_path):
        records, embedder, val_split, config = self._setup(tmp_path)
        trainer = Trainer(embedder, records, val_split, config, device="cpu")
        history = trainer.fit()

        assert len(history.epochs) == 2
        assert all(np.isfinite(epoch.loss) for epoch in history.epochs)
        assert history.best_epoch >= 1
        assert history.class_names == sorted({r.identity for r in records})

    def test_training_updates_the_parameters(self, tmp_path):
        records, embedder, val_split, config = self._setup(tmp_path)
        before = {k: v.detach().clone() for k, v in embedder.state_dict().items()}
        Trainer(embedder, records, val_split, config, device="cpu").fit()
        after = embedder.state_dict()
        changed = [k for k in before if not torch.allclose(before[k].float(), after[k].float())]
        assert changed, "optimiser stepped but no parameter changed"

    def test_best_loss_improves_over_training(self, tmp_path):
        """Assert on the *best* epoch, not a monotone sequence.

        Per-epoch means of a margin loss at scale 64 are noisy on a 24-image corpus, so
        demanding strict monotonicity would test the noise rather than the optimiser. What
        must hold is that the loop eventually finds a better loss than it started with.
        """
        records, embedder, val_split, config = self._setup(tmp_path)
        config.epochs = 6
        config.lr = 1e-2
        config.warmup_epochs = 1
        history = Trainer(embedder, records, val_split, config, device="cpu").fit()
        losses = [epoch.loss for epoch in history.epochs]
        assert min(losses[2:]) < losses[0] + 1e-9
        assert all(np.isfinite(value) for value in losses)

    def test_held_out_identities_are_retrievable_above_chance(self, tmp_path):
        """The end-to-end claim, stated as something that is actually stable.

        Measured first: on this 40-image synthetic corpus of solid-colour tiles, a *random*
        margin head already separates identities, so ``hit@1`` before and after training are
        both 1.000 in five out of five trials — there is no headroom for an improvement
        assertion to measure. A test asserting "training improves retrieval" therefore only
        passed when the random head happened to initialise badly, which made it flaky.

        What is both meaningful and stable: training must not destroy retrieval, and held-out
        identities must remain retrievable well above chance (1 / n_identities). The real
        accuracy claim is made in `docs/BENCHMARK.md` against the actual cat corpus.
        """
        records, embedder, val_split, config = self._setup(tmp_path, identities=8, per_identity=5)
        config.epochs = 6
        config.lr = 1e-2
        config.warmup_epochs = 1
        trainer = Trainer(embedder, records, val_split, config, device="cpu")
        before = trainer.validate()[0]
        trainer.fit(handle_signals=False)
        after = trainer.validate()[0]
        chance = 1.0 / len(set(val_split.query_labels.tolist()))
        assert after >= chance * 1.5, f"held-out retrieval {after:.3f} is near chance {chance:.3f}"
        assert after >= before - 0.15, "training degraded held-out retrieval substantially"

    def test_validation_reports_the_top1_hit_rate(self, tmp_path):
        """`validate` must return hit@1, not full recall.

        Full recall reads as roughly 1 / references-per-identity and barely moves with model
        quality, so using it for checkpoint selection silently picks the wrong epoch.
        """
        records, embedder, val_split, config = self._setup(tmp_path, identities=8, per_identity=5)
        embedder.eval()
        hit_at_1, map_at_5, identities = Trainer(
            embedder, records, val_split, config, device="cpu"
        ).validate()
        assert identities == 8
        # hit@1 is a share of queries, so it must land on a multiple of 1/8.
        assert abs(hit_at_1 * 8 - round(hit_at_1 * 8)) < 1e-6, (
            f"hit@1={hit_at_1} is not a multiple of 1/8, so it is probably full recall"
        )
        assert 0.0 <= map_at_5 <= 1.0

    def test_non_arcface_losses_are_accepted(self, tmp_path):
        for loss in ("cosface", "subcenter_arcface", "linear"):
            records, embedder, val_split, config = self._setup(tmp_path, identities=4, per_identity=3)
            embedder.head.kind = loss
            config.epochs = 1
            history = Trainer(embedder, records, val_split, config, device="cpu").fit()
            assert len(history.epochs) == 1
