"""Model-layer tests that do not need downloaded weights.

Everything here runs on CPU with randomly initialised modules. That is enough to test the
properties that a pretrained checkpoint would hide: shapes, normalisation, device
handling, gradient flow, checkpoint round-trips and the margin arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# ruff: noqa: E402 - these imports must follow importorskip, which would otherwise be
# reported as an unused import and the suite would fail rather than skip without torch.

from catface.errors import ArtifactError, ModelError
from catface.models.embedder import Embedder, EmbedderConfig
from catface.models.heads import MetricHead, TripletLoss, VarianceRegulariser
from catface.models.pooling import GeM, pool_tokens, pooled_dim, resolve_pooling


class TestPooling:
    def test_resolve_pooling_picks_gap_for_cnns(self):
        assert resolve_pooling("auto", has_class_token=False) == "gap"

    def test_resolve_pooling_picks_cls_gap_for_transformers(self):
        assert resolve_pooling("auto", has_class_token=True) == "cls_gap"

    def test_resolve_pooling_rejects_an_impossible_request(self):
        with pytest.raises(ModelError, match="class token"):
            resolve_pooling("cls", has_class_token=False)

    def test_pooled_dim_accounts_for_concatenation(self):
        assert pooled_dim("cls_gap", 768) == 1536
        assert pooled_dim("gap", 2048) == 2048

    def test_cls_uses_the_first_token(self):
        tokens = torch.randn(2, 5, 8)
        assert torch.allclose(pool_tokens(tokens, "cls", prefix_tokens=1), tokens[:, 0])

    def test_gap_averages_patch_tokens_only(self):
        tokens = torch.randn(2, 5, 8)
        pooled = pool_tokens(tokens, "gap", prefix_tokens=1)
        assert torch.allclose(pooled, tokens[:, 1:].mean(dim=1))

    def test_cls_gap_concatenates_to_double_width(self):
        tokens = torch.randn(2, 5, 8)
        assert pool_tokens(tokens, "cls_gap", prefix_tokens=1).shape == (2, 16)

    def test_already_pooled_input_passes_through(self):
        features = torch.randn(3, 7)
        assert pool_tokens(features, "gap").shape == (3, 7)

    def test_gem_requires_an_instance(self):
        with pytest.raises(ModelError, match="GeM"):
            pool_tokens(torch.randn(1, 4, 8), "gem", prefix_tokens=1)

    def test_gem_lies_between_mean_and_max(self):
        tokens = torch.rand(3, 16, 8) + 0.1
        gem_value = GeM(p=3.0)(tokens)
        mean_value = tokens.mean(dim=1)
        max_value = tokens.max(dim=1).values
        assert torch.all(gem_value >= mean_value - 1e-5)
        assert torch.all(gem_value <= max_value + 1e-5)

    def test_unknown_pooling_mode_is_rejected(self):
        with pytest.raises(ModelError, match="Unknown pooling"):
            resolve_pooling("median", has_class_token=True)


class TestMetricHead:
    def test_embedding_is_unit_norm(self):
        head = MetricHead(in_features=32, num_classes=5, embedding_dim=16)
        head.eval()
        embeddings = head.embed(torch.randn(4, 32))
        assert torch.allclose(embeddings.norm(dim=1), torch.ones(4), atol=1e-5)

    def test_output_width_follows_embedding_dim(self):
        head = MetricHead(in_features=32, num_classes=5, embedding_dim=11)
        assert head.embed(torch.randn(3, 32)).shape == (3, 11)

    def test_eval_mode_applies_no_margin(self):
        """At test time the margin must be inert, otherwise thresholds are miscalibrated."""
        head = MetricHead(in_features=16, num_classes=4, embedding_dim=16, kind="arcface", scale=10.0)
        features = torch.randn(3, 16)
        head.train(False)
        with_margin = head(features, torch.tensor([0, 1, 2]))
        without_labels = head(features, None)
        assert torch.allclose(with_margin, without_labels)

    def test_training_margin_reduces_the_target_logit(self):
        head = MetricHead(in_features=16, num_classes=4, embedding_dim=16,
                          kind="arcface", margin=0.5, scale=16.0)
        features = torch.randn(4, 16)
        labels = torch.tensor([0, 1, 2, 3])

        head.train(False)
        plain = head(features, labels)
        head.train(True)
        margined = head(features, labels)

        rows = torch.arange(4)
        assert torch.all(margined[rows, labels] < plain[rows, labels])

    def test_margin_scale_and_kind_validation(self):
        with pytest.raises(ModelError, match="margin"):
            MetricHead(in_features=8, num_classes=2, kind="arcface", margin=0.0)
        with pytest.raises(ModelError, match="margin"):
            MetricHead(in_features=8, num_classes=2, kind="arcface", margin=2.0)
        with pytest.raises(ModelError, match="scale"):
            MetricHead(in_features=8, num_classes=2, scale=0.0)
        with pytest.raises(ModelError, match="Unknown head"):
            MetricHead(in_features=8, num_classes=2, kind="softmax")

    @pytest.mark.parametrize("kind", ["arcface", "cosface", "subcenter_arcface", "linear"])
    def test_all_head_kinds_produce_finite_logits(self, kind):
        head = MetricHead(in_features=16, num_classes=8, embedding_dim=8, kind=kind)
        head.train(True)
        logits = head(torch.randn(6, 16), torch.tensor([0, 1, 2, 3, 4, 5]))
        assert logits.shape == (6, 8)
        assert torch.isfinite(logits).all()

    def test_subcenter_produces_one_logit_per_identity(self):
        head = MetricHead(in_features=16, num_classes=5, embedding_dim=8,
                          kind="subcenter_arcface", num_subcenters=3)
        assert head.module.weight.shape[0] == 15
        head.train(False)
        assert head(torch.randn(2, 16), None).shape == (2, 5)

    def test_inference_without_any_classes_returns_embeddings(self):
        head = MetricHead(in_features=16, num_classes=0, embedding_dim=8)
        assert head(torch.randn(2, 16)).shape == (2, 8)


class TestAuxiliaryLosses:
    def test_triplet_loss_is_zero_when_margins_are_satisfied(self):
        embeddings = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.98, 0.1], [0.0, 1.0]]), dim=1
        )
        labels = torch.tensor([0, 0, 1])
        assert float(TripletLoss(margin=0.1)(embeddings, labels)) == pytest.approx(0.0, abs=1e-6)

    def test_triplet_loss_is_positive_when_an_impostor_is_too_close(self):
        embeddings = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.9, 0.44], [0.95, 0.31]]), dim=1
        )
        labels = torch.tensor([0, 0, 1])
        assert float(TripletLoss(margin=0.3)(embeddings, labels)) > 0.0

    def test_triplet_loss_handles_a_batch_without_positive_pairs(self):
        embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
        loss = TripletLoss()(embeddings, torch.tensor([0, 1, 2]))
        assert torch.isfinite(loss)
        assert float(loss) == pytest.approx(0.0)

    def test_triplet_loss_is_differentiable(self):
        rng = torch.Generator().manual_seed(0)
        leaf = torch.randn(6, 4, generator=rng, requires_grad=True)
        embeddings = torch.nn.functional.normalize(leaf, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        TripletLoss()(embeddings, labels).backward()
        # Gradients flow to the leaf; the normalised tensor is a non-leaf by construction.
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()

    def test_variance_regulariser_is_inert_at_zero_weight(self):
        embeddings = torch.nn.functional.normalize(torch.randn(4, 4), dim=1)
        assert float(VarianceRegulariser(weight=0.0)(embeddings, torch.tensor([0, 0, 1, 1]))) == 0.0

    def test_variance_regulariser_penalises_a_spread_identity(self):
        tight = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.99, 0.1]]), dim=1)
        spread = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.1, 0.99]]), dim=1)
        labels = torch.tensor([0, 0])
        regulariser = VarianceRegulariser(weight=1.0)
        assert float(regulariser(spread, labels)) > float(regulariser(tight, labels))


class TestEmbedderMechanics:
    def _embedder(self, **overrides) -> Embedder:
        # A tiny timm model keeps this instant (no download, a few hundred thousand params).
        config = EmbedderConfig(
            backbone="timm",
            timm_name="resnet10t.c3_in1k",
            embedding_dim=32,
            **overrides,
        )
        return Embedder(config, num_classes=4, device="cpu")

    def test_embedding_shape_and_norm(self):
        embedder = self._embedder()
        embedder.eval()
        vectors = embedder.embed(torch.randn(2, 3, 64, 64))
        assert vectors.shape == (2, 32)
        assert torch.allclose(vectors.norm(dim=1), torch.ones(2), atol=1e-4)

    def test_embedding_is_deterministic_in_eval_mode(self):
        embedder = self._embedder()
        embedder.eval()
        images = torch.randn(1, 3, 64, 64)
        assert torch.allclose(embedder.embed(images), embedder.embed(images), atol=1e-6)

    def test_different_images_give_different_descriptors(self):
        embedder = self._embedder()
        embedder.eval()
        first = embedder.embed(torch.zeros(1, 3, 64, 64))
        second = embedder.embed(torch.ones(1, 3, 64, 64))
        assert not torch.allclose(first, second, atol=1e-3)

    def test_forward_returns_logits_with_the_class_count(self):
        embedder = self._embedder()
        logits = embedder(torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 2, 3]))
        assert logits.shape == (4, 4)

    def test_embedder_moves_inputs_to_its_device(self):
        """A CPU tensor must be accepted by a CPU model without an explicit .to()."""
        embedder = self._embedder()
        embedder.to("cpu")
        assert embedder.embed(torch.randn(1, 3, 64, 64)).shape[0] == 1

    def test_parameter_groups_scale_the_backbone_learning_rate(self):
        embedder = self._embedder()
        groups = embedder.named_parameter_groups(lr=1e-3, backbone_lr_scale=0.1, weight_decay=0.05)
        by_name = {group["name"]: group for group in groups}
        assert "backbone" in by_name
        assert by_name["backbone"]["lr"] == pytest.approx(1e-4)
        assert all(group["lr"] <= 1e-3 for group in groups if group["name"] != "backbone")

    def test_head_parameters_are_present_in_the_optimiser_groups(self):
        embedder = self._embedder()
        groups = embedder.named_parameter_groups(lr=1e-3)
        grouped_ids = {id(p) for group in groups for p in group["params"]}
        head_ids = {id(p) for p in embedder.head.parameters()}
        assert head_ids <= grouped_ids

    def test_checkpoint_round_trip_preserves_descriptors(self, tmp_path):
        embedder = self._embedder()
        embedder.eval()
        images = torch.randn(1, 3, 64, 64)
        before = embedder.embed(images).detach().numpy()

        path = embedder.save(tmp_path / "model.pt", extra={"note": "unit test"})
        restored = Embedder.load(path, device="cpu")
        after = restored.embed(images).detach().numpy()

        assert np.allclose(before, after, atol=1e-5)
        assert restored.config.embedding_dim == 32

    def test_loading_a_non_checkpoint_is_reported(self, tmp_path):
        torch.save({"unrelated": 1}, tmp_path / "junk.pt")
        with pytest.raises(ArtifactError, match="not a catface checkpoint"):
            Embedder.load(tmp_path / "junk.pt")

    def test_loading_a_missing_checkpoint_is_reported(self, tmp_path):
        with pytest.raises(ArtifactError, match="not found"):
            Embedder.load(tmp_path / "absent.pt")

    def test_checkpoint_version_mismatch_is_reported(self, tmp_path):
        embedder = self._embedder()
        path = embedder.save(tmp_path / "model.pt")
        payload = torch.load(path, weights_only=False)
        payload["version"] = 999
        torch.save(payload, path)
        with pytest.raises(ArtifactError, match="version"):
            Embedder.load(path)

    def test_describe_config_reports_the_pieces_that_matter(self):
        described = self._embedder().describe_config()
        for key in ("backbone", "embedding_dim", "pooling", "head", "image_size"):
            assert key in described

    def test_auto_pooling_resolves_differently_for_cnn_and_transformer(self):
        cnn = self._embedder(pooling="auto")
        assert cnn.pooling == "gap"
        transformer = Embedder(
            EmbedderConfig(backbone="timm", timm_name="vit_tiny_patch16_224.augreg_in21k",
                           embedding_dim=16, pooling="auto"),
            num_classes=0, device="cpu",
        )
        assert transformer.pooling == "cls_gap"


class TestTTATransforms:
    def test_views_produce_the_expected_count_and_shape(self):
        from catface.models.embedder import build_tta_transforms

        transforms = build_tta_transforms(64, ("identity", "hflip"))
        assert len(transforms) == 2
        from PIL import Image

        image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        for _, transform in transforms:
            assert transform(image).shape == (3, 64, 64)

    def test_hflip_view_differs_from_the_identity_view(self):
        from PIL import Image

        from catface.models.embedder import build_tta_transforms

        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 255, (80, 80, 3), dtype=np.uint8))
        transforms = dict(build_tta_transforms(64, ("identity", "hflip")))
        assert not torch.allclose(transforms["identity"](image), transforms["hflip"](image))

    def test_unknown_view_is_rejected(self):
        from catface.models.embedder import build_tta_transforms

        with pytest.raises(ModelError, match="Unknown TTA view"):
            build_tta_transforms(64, ("identity", "rotate"))
