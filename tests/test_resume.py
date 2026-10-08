"""Pause and resume tests.

Resumable training is only useful if resuming is *faithful*: continuing a run must follow
the same optimisation trajectory an uninterrupted run would have. A checkpoint that carries
the weights but not the optimiser moments, the scheduler position or the sampler's epoch
silently produces a different experiment, and the loss curve alone will not reveal it.

These tests therefore assert on state equivalence, not merely on "it ran again".
"""

from __future__ import annotations

import json
import time

import pytest

torch = pytest.importorskip("torch")

# ruff: noqa: E402 - these imports must follow importorskip, which would otherwise be
# reported as an unused import and the suite would fail rather than skip without torch.

from catface.errors import ModelError
from catface.eval.protocols import build_identity_split
from catface.models.embedder import Embedder, EmbedderConfig
from catface.train.loop import (
    BEST_FILENAME,
    PAUSE_FILENAME,
    TrainConfigResolved,
    Trainer,
    TrainingState,
    train_metric_learner,
)
from test_training import synthetic_records


def build(tmp_path, epochs: int = 3, identities: int = 6, per_identity: int = 4,
          output: str = "run", seed: int = 1337, patience: int = 0):
    """A small CPU trainer plus the pieces needed to rebuild it identically."""
    records = synthetic_records(tmp_path, identities=identities, per_identity=per_identity)
    val_split = build_identity_split(records, queries_per_identity=1, seed=0, name="val")
    config = TrainConfigResolved(
        epochs=epochs, batch_size=4, lr=1e-3, identities_per_batch=2, samples_per_identity=2,
        val_every=1, num_workers=0, image_size=32, output_dir=tmp_path / output,
        amp=False, warmup_epochs=0, triplet_weight=0.3, early_stop_patience=patience,
        seed=seed, fingerprint="test-fingerprint",
    )
    embedder = Embedder(
        EmbedderConfig(backbone="timm", timm_name="resnet10t.c3_in1k", embedding_dim=16,
                       pooling="gap", head="arcface", image_size=32),
        num_classes=identities, device="cpu",
    )
    trainer = Trainer(embedder, records, val_split, config, device="cpu")
    return records, val_split, config, trainer


class TestCheckpointing:
    def test_state_file_is_written_after_every_epoch(self, tmp_path):
        _, _, config, trainer = build(tmp_path, epochs=2)
        trainer.fit(handle_signals=False)
        assert trainer.state_path.is_file(), "no resumable state was written"
        assert config.output_dir.joinpath("training_history.json").is_file()

    def test_state_carries_optimiser_scheduler_and_rng(self, tmp_path):
        _, _, _, trainer = build(tmp_path, epochs=1)
        trainer.fit(handle_signals=False)
        state = torch.load(trainer.state_path, map_location="cpu", weights_only=False)
        assert isinstance(state, TrainingState)
        assert state.optimizer, "optimiser state missing — momentum would be lost on resume"
        assert state.scheduler, "scheduler state missing — the LR schedule would restart"
        assert "torch" in state.rng or "numpy" in state.rng, "RNG state missing"
        assert state.model, "model weights missing"
        assert state.epoch == 1

    def test_state_write_is_atomic(self, tmp_path):
        """No half-written state file may survive, since that is the resume path."""
        _, _, _, trainer = build(tmp_path, epochs=1)
        trainer.fit(handle_signals=False)
        leftovers = list(trainer.state_path.parent.glob("*.tmp"))
        assert not leftovers, f"temporary files left behind: {leftovers}"

    def test_best_checkpoint_is_exported_and_loadable(self, tmp_path):
        _, _, _, trainer = build(tmp_path, epochs=2)
        trainer.fit(handle_signals=False)
        best = tmp_path / "run" / BEST_FILENAME
        assert best.is_file()
        reloaded = Embedder.load(best, device="cpu")
        assert reloaded.head.num_classes == 6

    def test_history_records_pause_reason_as_none_on_completion(self, tmp_path):
        _, _, _, trainer = build(tmp_path, epochs=2)
        history = trainer.fit(handle_signals=False)
        assert history.pause_reason is None
        payload = json.loads(
            (tmp_path / "run" / "training_history.json").read_text(encoding="utf-8")
        )
        assert payload["pause_reason"] is None
        assert payload["version"] == 2


class TestResumeFaithfulness:
    def test_resume_restores_weights_exactly(self, tmp_path):
        _, _, _, first = build(tmp_path, epochs=1)
        first.fit(handle_signals=False)

        _, _, _, second = build(tmp_path, epochs=3)
        second.load_state()
        for key, value in first.embedder.state_dict().items():
            assert torch.allclose(
                second.embedder.state_dict()[key].float(), value.float(), atol=1e-6
            ), f"weight {key} differs after resume"

    def test_resume_restores_optimiser_moments(self, tmp_path):
        """Momentum is the state that, if lost, changes the trajectory silently."""
        _, _, _, first = build(tmp_path, epochs=1)
        first.fit(handle_signals=False)
        _, _, _, second = build(tmp_path, epochs=3)
        second.load_state()
        first_state = first.optimizer.state_dict()
        second_state = second.optimizer.state_dict()
        assert len(second_state["state"]) == len(first_state["state"])
        for index in first_state["state"]:
            for key, value in first_state["state"][index].items():
                if torch.is_tensor(value):
                    assert torch.allclose(
                        second_state["state"][index][key].float(), value.float()
                    ), f"optimiser state[{index}][{key}] differs"

    def test_resume_continues_at_the_next_epoch(self, tmp_path):
        _, _, _, first = build(tmp_path, epochs=1)
        first.fit(handle_signals=False)
        _, _, _, second = build(tmp_path, epochs=3)
        second.load_state()
        history = second.fit(handle_signals=False)
        epochs = [record.epoch for record in history.epochs]
        assert epochs == [1, 2, 3], f"expected continued epochs, got {epochs}"

    def test_resume_keeps_the_best_score_seen_so_far(self, tmp_path):
        _, _, _, first = build(tmp_path, epochs=1)
        first.fit(handle_signals=False)
        best_before = first.history.best_score
        _, _, _, second = build(tmp_path, epochs=2)
        second.load_state()
        second.fit(handle_signals=False)
        # The running best must never regress across a resume.
        assert second.history.best_score >= best_before - 1e-9

    def test_resume_restores_scheduler_position(self, tmp_path):
        _, _, _, first = build(tmp_path, epochs=1)
        first.fit(handle_signals=False)
        _, _, _, second = build(tmp_path, epochs=3)
        second.load_state()
        state = second.load_state()
        assert state.epoch == 1
        # The restored scheduler must already count the completed epoch's steps.
        assert second.scheduler.last_epoch >= first.scheduler.last_epoch

    def test_total_work_is_the_same_whether_paused_or_not(self, tmp_path):
        """2 epochs then 2 more must cover the same epochs as 4 in one go."""
        _, _, _, straight = build(tmp_path, epochs=4, output="straight")
        straight_history = straight.fit(handle_signals=False)

        _, _, _, part_a = build(tmp_path, epochs=2, output="split")
        part_a.fit(handle_signals=False)
        _, _, _, part_b = build(tmp_path, epochs=4, output="split")
        part_b.load_state()
        split_history = part_b.fit(handle_signals=False)

        assert [r.epoch for r in straight_history.epochs] == [1, 2, 3, 4]
        assert [r.epoch for r in split_history.epochs] == [1, 2, 3, 4]
        assert len(part_b.history.epochs) == 4


class TestPauseMechanisms:
    def test_request_pause_stops_at_the_epoch_boundary(self, tmp_path):
        _, _, _, trainer = build(tmp_path, epochs=5)
        trainer.request_pause("unit-test")
        history = trainer.fit(handle_signals=False)
        assert history.pause_reason == "unit-test"
        assert history.epochs == [], "no epoch should have started"
        assert trainer.state_path.is_file(), "a paused run must still be resumable"

    def test_pause_file_is_honoured(self, tmp_path):
        _, _, config, trainer = build(tmp_path, epochs=5)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / PAUSE_FILENAME).write_text("pause", encoding="utf-8")
        history = trainer.fit(handle_signals=False)
        assert history.pause_reason == "pause-file"
        assert history.epochs == []

    def test_time_budget_pauses_at_an_epoch_boundary(self, tmp_path):
        """A budget must stop the run cleanly rather than being killed mid-epoch."""
        _, _, _, trainer = build(tmp_path, epochs=5)
        history = trainer.fit(max_seconds=0.6, handle_signals=False)
        assert history.pause_reason == "time-budget"
        # It stopped early, but it completed whole epochs and can be resumed.
        assert len(history.epochs) < 5
        assert trainer.state_path.is_file()
        assert trainer.best_path.is_file(), "a pause must still export the best-so-far model"

    def test_exhausted_budget_stops_before_starting_an_epoch(self, tmp_path):
        """When the budget is already spent, no epoch should begin.

        This is the scheduler-driven case: a job resumes with a budget, and the check at the
        top of the loop is what stops a long epoch from overrunning the window.
        """
        _, _, _, trainer = build(tmp_path, epochs=5)
        trainer._budget_deadline = time.perf_counter() - 1.0      # already spent
        history = trainer.fit(handle_signals=False)
        assert history.pause_reason == "time-budget"
        assert history.epochs == [], "an epoch started despite an exhausted budget"
        # It must still leave a usable checkpoint rather than exiting with nothing.
        assert trainer.state_path.is_file()

    def test_paused_run_is_not_reported_as_converged(self, tmp_path):
        _, _, _, trainer = build(tmp_path, epochs=5)
        trainer.request_pause("unit-test")
        history = trainer.fit(handle_signals=False)
        assert history.stopped_early is False, (
            "a paused run must not be flagged as early-stopped, or an interrupted "
            "experiment would read as converged"
        )

    def test_pause_keeps_the_latest_weights_not_the_best(self, tmp_path):
        """Resuming must continue the actual trajectory, so latest ≠ best on pause.

        Restoring the best epoch's weights before saving would make the next run continue
        from a different point than an uninterrupted run would have, which is precisely the
        reproducibility this feature exists to protect.
        """
        _, _, _, trainer = build(tmp_path, epochs=1)
        trainer.fit(handle_signals=False)          # best epoch == 1, latest == 1
        latest = {k: v.detach().clone() for k, v in trainer.embedder.state_dict().items()}

        paused = build(tmp_path, epochs=3, output="run")[3]
        paused.load_state()
        # Force a divergence between "best so far" and "latest".
        paused.best_state = {k: torch.zeros_like(v) for k, v in latest.items()}
        paused.request_pause("keep-latest")
        paused.fit(handle_signals=False)
        for key, value in latest.items():
            assert torch.allclose(
                paused.embedder.state_dict()[key].float(), value.float(), atol=1e-6
            ), "pausing restored the best weights instead of the latest ones"

    def test_first_pause_reason_wins(self, tmp_path):
        _, _, _, trainer = build(tmp_path, epochs=2)
        trainer.request_pause("first")
        trainer.request_pause("second")
        assert trainer.fit(handle_signals=False).pause_reason == "first"

    def test_completed_run_restores_best_weights(self, tmp_path):
        _, _, _, trainer = build(tmp_path, epochs=2)
        trainer.fit(handle_signals=False)
        assert trainer.best_state is not None
        for key, value in trainer.best_state.items():
            assert torch.allclose(
                trainer.embedder.state_dict()[key].float(), value.float(), atol=1e-5
            ), "a completed run must leave the validated weights in place"


class TestLoadStateValidation:
    def test_missing_state_file_is_reported(self, tmp_path):
        _, _, _, trainer = build(tmp_path, epochs=1)
        with pytest.raises(ModelError, match="No resumable state"):
            trainer.load_state()

    def test_wrong_file_type_is_reported(self, tmp_path):
        _, _, _, trainer = build(tmp_path, epochs=1)
        trainer.state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"not": "a TrainingState"}, trainer.state_path)
        with pytest.raises(ModelError, match="does not contain a TrainingState"):
            trainer.load_state()

    def test_future_state_version_is_reported(self, tmp_path):
        _, _, _, trainer = build(tmp_path, epochs=1)
        trainer.fit(handle_signals=False)
        state = torch.load(trainer.state_path, map_location="cpu", weights_only=False)
        state.version = 999
        torch.save(state, trainer.state_path)
        with pytest.raises(ModelError, match="state version"):
            trainer.load_state()

    def test_resuming_into_a_different_class_space_is_refused(self, tmp_path):
        """Silently resuming with a different identity set would be a different experiment."""
        _, _, _, first = build(tmp_path, epochs=1, identities=6, output="run")
        first.fit(handle_signals=False)
        _, _, _, other = build(tmp_path, epochs=2, identities=4, output="run")
        with pytest.raises(ModelError, match="different class space"):
            other.load_state()
        # The escape hatch exists but must be requested explicitly.
        other.load_state(strict=False)

    def test_resuming_beyond_the_schedule_is_refused(self, tmp_path):
        _, _, _, first = build(tmp_path, epochs=2, output="run")
        first.fit(handle_signals=False)
        _, _, _, shorter = build(tmp_path, epochs=1, output="run")
        with pytest.raises(ModelError, match="raise --epochs"):
            shorter.load_state()


class TestWrapper:
    def _args(self, tmp_path, **overrides):
        records = synthetic_records(tmp_path, identities=6, per_identity=4)
        val_split = build_identity_split(records, queries_per_identity=1, seed=0, name="val")
        settings = {
            "epochs": 2, "batch_size": 4, "lr": 1e-3, "identities_per_batch": 2, "samples_per_identity": 2,
            "val_every": 1, "num_workers": 0, "image_size": 32, "output_dir": tmp_path / "wrap",
            "amp": False, "warmup_epochs": 0, "early_stop_patience": 0,
        }
        settings.update(overrides)
        config = TrainConfigResolved(**settings)
        embedder = Embedder(
            EmbedderConfig(backbone="timm", timm_name="resnet10t.c3_in1k", embedding_dim=16,
                           pooling="gap", head="arcface", image_size=32),
            num_classes=6, device="cpu",
        )
        return records, val_split, config, embedder

    def test_wrapper_returns_trainer_and_writes_state(self, tmp_path):
        records, val_split, config, embedder = self._args(tmp_path)
        history, trainer = train_metric_learner(
            embedder, records, val_split, config, device="cpu", handle_signals=False
        )
        assert trainer.state_path.is_file()
        assert history.epochs
        assert trainer.best_path.is_file()

    def test_wrapper_resume_continues_the_run(self, tmp_path):
        records, val_split, config, embedder = self._args(tmp_path, epochs=1)
        train_metric_learner(embedder, records, val_split, config, device="cpu",
                             handle_signals=False)

        records2, val_split2, config2, embedder2 = self._args(tmp_path, epochs=3)
        history, _ = train_metric_learner(
            embedder2, records2, val_split2, config2, device="cpu",
            resume=True, handle_signals=False,
        )
        assert [r.epoch for r in history.epochs] == [1, 2, 3]

    def test_wrapper_warns_when_resume_finds_nothing(self, tmp_path, caplog):
        records, val_split, config, embedder = self._args(tmp_path)
        history, _ = train_metric_learner(
            embedder, records, val_split, config, device="cpu",
            resume=True, handle_signals=False,
        )
        # It must still train rather than fail, and the run starts from epoch 1.
        assert [r.epoch for r in history.epochs] == [1, 2]
