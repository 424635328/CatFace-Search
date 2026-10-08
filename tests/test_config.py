"""Configuration loading, validation and reproducibility tests."""

from __future__ import annotations

import pytest

from catface.config import (
    DataConfig,
    EvalConfig,
    ModelConfig,
    PipelineConfig,
    TrainConfig,
    dump_config,
    load_config,
)
from catface.errors import ConfigError

CONFIG_FILE = "configs/default.yaml"


class TestDefaults:
    def test_defaults_construct_without_a_file(self):
        config = PipelineConfig()
        assert config.model.backbone
        assert config.data.tile > 0
        assert config.eval.recall_ks[0] == 1

    def test_paths_are_absolute(self):
        config = PipelineConfig()
        assert config.data.root.is_absolute()
        assert config.data.crops_dir.is_absolute()
        assert config.train.output_dir.is_absolute()


class TestValidation:
    def test_unknown_key_is_rejected(self):
        """A typo in a config file must fail loudly, not be silently ignored."""
        with pytest.raises(ConfigError, match="Unknown key"):
            PipelineConfig.from_mapping({"model": {"backbone": "resnet50", "backbne": "x"}})

    def test_unknown_top_level_key_is_rejected(self):
        with pytest.raises(ConfigError, match="Unknown key"):
            PipelineConfig.from_mapping({"models": {}})

    def test_invalid_backbone_is_rejected(self):
        with pytest.raises(ConfigError, match="backbone"):
            ModelConfig(backbone="resnet5000")

    def test_timm_backbone_requires_a_name(self):
        with pytest.raises(ConfigError, match="timm_name"):
            ModelConfig(backbone="timm")

    def test_invalid_head_and_pooling_are_rejected(self):
        with pytest.raises(ConfigError, match="pooling"):
            ModelConfig(pooling="avg")
        with pytest.raises(ConfigError, match="head"):
            ModelConfig(head="softmax")

    def test_invalid_loss_is_rejected(self):
        with pytest.raises(ConfigError, match="loss"):
            TrainConfig(loss="magic")

    def test_negative_epochs_rejected(self):
        with pytest.raises(ConfigError, match="epochs"):
            TrainConfig(epochs=0)

    def test_whiten_and_qe_values_are_constrained(self):
        with pytest.raises(ConfigError, match="whiten"):
            EvalConfig(whiten="zca")
        with pytest.raises(ConfigError, match="query_expansion"):
            EvalConfig(query_expansion="rqe")

    def test_recall_ks_must_be_positive(self):
        with pytest.raises(ConfigError, match="recall_ks"):
            EvalConfig(recall_ks=(0, 5))

    def test_pad_ratio_bounds(self):
        with pytest.raises(ConfigError, match="pad_ratio"):
            DataConfig(pad_ratio=5.0)

    def test_alignment_value_is_constrained(self):
        with pytest.raises(ConfigError, match="alignment"):
            DataConfig(alignment="magic")


class TestOverrides:
    def test_dotted_override_applies(self):
        config = load_config(None, {"model.backbone": "resnet50", "train.epochs": 3})
        assert config.model.backbone == "resnet50"
        assert config.train.epochs == 3

    def test_override_on_unknown_key_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown key"):
            load_config(None, {"model.nonexistent": 1})

    def test_override_through_non_section_is_rejected(self):
        with pytest.raises(ConfigError, match="not a section"):
            load_config(None, {"model.backbone.deep": 1})

    def test_overrides_are_revalidated(self):
        """An override that violates a constraint must still fail."""
        with pytest.raises(ConfigError):
            load_config(None, {"model.embedding_dim": -4})


class TestFingerprint:
    def test_fingerprint_is_stable(self):
        a = PipelineConfig()
        b = PipelineConfig()
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_changes_with_semantics(self):
        a = PipelineConfig()
        b = PipelineConfig.from_mapping({**a.to_mapping(), "model": {**a.model.to_dict(), "backbone": "resnet50"}})
        assert a.fingerprint() != b.fingerprint()

    def test_fingerprint_ignores_output_paths(self):
        """Two runs with the same recipe must share a fingerprint even in different dirs."""
        a = PipelineConfig(output_dir="artifacts/one")
        b = PipelineConfig(output_dir="artifacts/two")
        assert a.fingerprint() == b.fingerprint()


class TestRoundTrip:
    def test_dump_and_reload_preserves_values(self, tmp_path):
        original = PipelineConfig(
            run_name="round-trip",
            data=DataConfig(tile=192, pad_ratio=0.2),
            model=ModelConfig(backbone="resnet50", embedding_dim=256),
            train=TrainConfig(epochs=4, lr=1e-3),
            eval=EvalConfig(recall_ks=(1, 3)),
        )
        path = dump_config(original, tmp_path / "config.yaml")
        reloaded = PipelineConfig.from_yaml(path)

        assert reloaded.run_name == "round-trip"
        assert reloaded.data.tile == 192
        assert reloaded.model.backbone == "resnet50"
        assert reloaded.model.embedding_dim == 256
        assert reloaded.train.epochs == 4
        assert reloaded.eval.recall_ks == (1, 3)
        assert reloaded.fingerprint() == original.fingerprint()

    def test_shipped_default_config_is_valid(self):
        """The checked-in reference config must parse — it backs the reported numbers."""
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[1] / CONFIG_FILE
        if not config_path.is_file():
            pytest.skip(f"{CONFIG_FILE} not present")
        config = PipelineConfig.from_yaml(config_path)
        assert config.model.backbone
        assert config.train.epochs >= 1
        assert 1 in config.eval.recall_ks

    def test_non_mapping_yaml_is_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            PipelineConfig.from_yaml(path)

    def test_empty_yaml_yields_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        config = PipelineConfig.from_yaml(path)
        assert config.model.backbone == PipelineConfig().model.backbone

    def test_missing_file_is_reported(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            PipelineConfig.from_yaml(tmp_path / "nope.yaml")
