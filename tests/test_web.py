"""Web layer tests.

The service layer is exercised with a deliberately fake embedder so the suite stays fast and
deterministic: the questions worth testing here are protocol questions (what is refused, what is
reported, does a near tie stay visible), and none of them need real ViT weights.

FastAPI is an optional dependency group. When it is absent these tests skip rather than fail, so a
core-only installation still gets a green suite; CI installs the ``web`` extra so the HTTP contract
is genuinely exercised there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from catface.data.manifest import Manifest
from catface.errors import DataError
from catface.web.service import MAX_UPLOAD_BYTES, QueryRejectedError, SearchService

pytest.importorskip("fastapi", reason="the web extra is not installed")

from fastapi.testclient import TestClient

from catface.web.api import LIMITATIONS, create_app


def _manifest_records(path):
    """Read manifest records through the real loader, so the records have the real type."""
    return Manifest.load(path)


# ------------------------------------------------------------------------------------------------
# a service that does not need a model
# ------------------------------------------------------------------------------------------------
class FakeEmbedder:
    """Returns a fixed descriptor per file, derived from the filename.

    Deterministic by construction: the test needs to know which gallery entry is nearest, not to
    exercise the backbone.
    """

    class _Config:
        backbone = "fake-backbone"
        tta: ClassVar[list[str]] = ["identity", "hflip"]

    def __init__(self, mapping: dict[str, np.ndarray] | None = None) -> None:
        self.config = self._Config()
        self.mapping = mapping or {}
        self.seen: list[str] = []

    def describe_config(self) -> dict[str, str]:
        return {"backbone": self.config.backbone}


class FakeResult:
    def __init__(self, vectors: np.ndarray, ids: list[str]) -> None:
        self.vectors = vectors
        self.ids = ids


def make_service(tmp_path: Path, vectors: dict[str, list[float]], labels: dict[str, str]) -> SearchService:
    """A SearchService whose model load and embedding are stubbed out."""
    manifest = tmp_path / "manifest.jsonl"
    lines = []
    for image_id, label in labels.items():
        image = tmp_path / f"{image_id}.jpg"
        image.write_bytes(b"fake")
        lines.append(
            json.dumps(
                {
                    "image_id": image_id,
                    "identity": label,
                    "path": str(image),
                    "source": "test",
                    "detector": "whole_image",
                }
            )
        )
    manifest.write_text("\n".join(lines), encoding="utf-8")

    service = SearchService(checkpoint=tmp_path / "unused.pt", manifest=manifest, device="cpu")
    matrix = np.asarray(list(vectors.values()), dtype=np.float32)
    service._embedder = FakeEmbedder()
    service._vectors = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    service._records = list(_manifest_records(manifest))
    return service


def stub_embed(service: SearchService, vector: list[float], monkeypatch) -> None:
    """Make every embed call return ``vector`` regardless of the input path."""
    import catface.web.service as module

    array = np.asarray([vector], dtype=np.float32)

    def fake_embed_records(embedder, paths, **kwargs):
        return FakeResult(array, ["query"])

    monkeypatch.setattr(module, "embed_records", fake_embed_records)


# ------------------------------------------------------------------------------------------------
# service behaviour
# ------------------------------------------------------------------------------------------------
class TestServiceRanking:
    def test_nearest_gallery_entry_is_returned_first(self, tmp_path, monkeypatch):
        service = make_service(
            tmp_path,
            vectors={"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.9, 0.1]},
            labels={"a": "cat-a", "b": "cat-b", "c": "cat-a"},
        )
        stub_embed(service, [1.0, 0.0], monkeypatch)
        query = tmp_path / "query.jpg"
        query.write_bytes(b"fake")

        outcome = service.search(query, top_k=3)
        assert outcome.predicted_identity == "cat-a"
        assert outcome.matches[0].image_id == "a"
        assert outcome.matches[0].rank == 1
        assert outcome.descriptor_dim == 2

    def test_margin_reports_the_gap_to_the_best_different_identity(self, tmp_path, monkeypatch):
        """A near tie must be visible in the answer, because every measured failure was one."""
        service = make_service(
            tmp_path,
            vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]},
            labels={"a": "cat-a", "b": "cat-b"},
        )
        stub_embed(service, [0.5, 0.5], monkeypatch)
        query = tmp_path / "query.jpg"
        query.write_bytes(b"fake")

        outcome = service.search(query, top_k=2)
        # Both identities sit at cosine 0.707, so the margin must be ~0 rather than large.
        assert outcome.margin is not None
        assert abs(outcome.margin) < 1e-5, "an exact tie must not be reported as a confident win"

    def test_identity_aggregation_can_only_lower_an_identity_score(self, tmp_path, monkeypatch):
        """The documented rule is the top-2 mean over one identity's gallery images.

        The property that matters is not "which identity wins" — it is that the pooled score of an
        identity is never above that identity's own best image. Pooling is therefore incapable of
        promoting an identity, and can only reshuffle near ties. This was measured on the real
        benchmark (503 queries, zero monotonicity violations) after a first version of this test
        asserted a false premise: a rule that averages can never beat the maximum it averages.
        """
        service = make_service(
            tmp_path,
            vectors={"a": [1.0, 0.0], "b": [0.98, 0.02], "c": [0.90, 0.40]},
            labels={"a": "cat-a", "b": "cat-a", "c": "cat-b"},
        )
        stub_embed(service, [1.0, 0.0], monkeypatch)
        query = tmp_path / "query.jpg"
        query.write_bytes(b"fake")

        single = service.search(query, top_k=3, identity_aggregation=False)
        pooled = service.search(query, top_k=3, identity_aggregation=True)

        best_cat_a = max(match.similarity for match in single.matches if match.identity == "cat-a")
        pooled_cat_a = next(match.similarity for match in pooled.matches if match.identity == "cat-a")
        assert pooled_cat_a <= best_cat_a + 1e-6, (
            "pooling must never raise an identity above its own best image; if this fails the "
            "aggregation is not a mean of that identity's similarities"
        )
        # The returned representative image is that identity's most similar image under both rules.
        assert single.matches[0].image_id == pooled.matches[0].image_id


class TestServiceRefusals:
    def test_missing_query_file_is_rejected(self, tmp_path, monkeypatch):
        service = make_service(tmp_path, vectors={"a": [1.0, 0.0]}, labels={"a": "cat-a"})
        stub_embed(service, [1.0, 0.0], monkeypatch)
        with pytest.raises(QueryRejectedError):
            service.search(tmp_path / "does-not-exist.jpg")

    def test_degenerate_descriptor_is_rejected(self, tmp_path, monkeypatch):
        """A zero descriptor would make every cosine meaningless; it must not be scored."""
        service = make_service(tmp_path, vectors={"a": [1.0, 0.0]}, labels={"a": "cat-a"})
        stub_embed(service, [0.0, 0.0], monkeypatch)
        query = tmp_path / "query.jpg"
        query.write_bytes(b"fake")
        with pytest.raises(QueryRejectedError):
            service.search(query)

    def test_empty_embedding_is_rejected(self, tmp_path, monkeypatch):
        import catface.web.service as module

        service = make_service(tmp_path, vectors={"a": [1.0, 0.0]}, labels={"a": "cat-a"})
        monkeypatch.setattr(
            module,
            "embed_records",
            lambda *a, **k: FakeResult(np.empty((0, 0), dtype=np.float32), []),
        )
        query = tmp_path / "query.jpg"
        query.write_bytes(b"fake")
        with pytest.raises(QueryRejectedError):
            service.search(query)

    def test_missing_checkpoint_names_the_fix(self, tmp_path):
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("", encoding="utf-8")
        service = SearchService(checkpoint=tmp_path / "nope.pt", manifest=manifest)
        with pytest.raises(DataError) as info:
            service.load()
        message = str(info.value)
        assert "checkpoint not found" in message
        assert "train_embedder" in message, "the error must say how to produce the missing file"


class TestServiceStatus:
    def test_status_reports_what_an_answer_depends_on(self, tmp_path):
        service = make_service(
            tmp_path,
            vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]},
            labels={"a": "cat-a", "b": "cat-a"},
        )
        status = service.status()
        assert status["ready"] is True
        assert status["gallery_images"] == 2
        assert status["gallery_identities"] == 1, "two images of one identity"
        assert status["backend"] == "fake-backbone"
        assert status["tta"] == ["identity", "hflip"]

    def test_identity_index_counts_images_per_identity(self, tmp_path):
        service = make_service(
            tmp_path,
            vectors={"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
            labels={"a": "cat-a", "b": "cat-a", "c": "cat-b"},
        )
        assert service.identity_index() == {"cat-a": 2, "cat-b": 1}


# ------------------------------------------------------------------------------------------------
# HTTP contract
# ------------------------------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    service = make_service(
        tmp_path,
        vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]},
        labels={"a": "cat-a", "b": "cat-b"},
    )
    stub_embed(service, [1.0, 0.0], monkeypatch)
    with TestClient(create_app(service)) as test_client:
        yield test_client


def upload(client, name: str = "query.jpg", payload: bytes = b"fake"):
    return client.post("/api/search", files={"file": (name, payload, "image/jpeg")}, data={"top_k": "5"})


class TestHttp:
    def test_healthz_does_not_depend_on_the_model(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_status_exposes_the_model_contract(self, client):
        payload = client.get("/api/status").json()
        assert payload["ready"] is True
        assert payload["gallery_images"] == 2
        assert payload["descriptor_dim"] == 2
        assert "version" in payload

    def test_search_returns_the_ranked_answer(self, client):
        response = upload(client)
        assert response.status_code == 200
        payload = response.json()
        assert payload["predicted_identity"] == "cat-a"
        assert payload["matches"][0]["image_id"] == "a"
        assert payload["timing"]["search_ms"] >= 0
        assert payload["descriptor_dim"] == 2

    def test_openapi_schema_is_generated(self, client):
        """The contract is the schema; if it cannot be generated, clients cannot rely on it."""
        schema = client.get("/openapi.json").json()
        assert "/api/search" in schema["paths"]
        assert "SearchResponse" in schema["components"]["schemas"]

    @pytest.mark.parametrize("name", ["query.gif", "query.txt", "query", "query.svg"])
    def test_unsupported_extension_is_refused_with_the_list(self, client, name):
        response = upload(client, name=name)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "accepted" in detail, "the refusal must say what is accepted"

    def test_empty_upload_is_refused(self, client):
        response = upload(client, payload=b"")
        assert response.status_code == 400
        assert "empty" in response.json()["detail"]

    def test_oversized_upload_is_refused_with_the_limit(self, client, monkeypatch):
        import catface.web.api as module

        monkeypatch.setattr(module, "MAX_UPLOAD_BYTES", 16)
        response = upload(client, payload=b"x" * 64)
        assert response.status_code == 400
        assert "limit" in response.json()["detail"]

    def test_gallery_image_is_served(self, client):
        response = client.get("/api/gallery/a")
        assert response.status_code == 200

    def test_unknown_gallery_image_is_404_not_500(self, client):
        response = client.get("/api/gallery/nope")
        assert response.status_code == 404
        assert "no gallery entry" in response.json()["detail"]

    def test_index_page_renders_the_limitations(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.text
        for item in LIMITATIONS:
            assert item["title"] in body, "the UI must state its measured limits, not hide them"
        assert "cat-a" not in body, "the page must not leak gallery labels before a search"

    def test_missing_service_reports_503_not_500(self):
        """A deployment without a model must look unready, not broken."""
        with TestClient(create_app(None)) as bare:
            assert bare.get("/healthz").status_code == 200
            response = bare.post("/api/search", files={"file": ("q.jpg", b"x", "image/jpeg")})
            assert response.status_code == 503

    def test_api_key_is_enforced_when_configured(self, client, monkeypatch):
        monkeypatch.setenv("CATFACE_API_KEY", "secret")
        assert upload(client).status_code == 401
        assert client.get("/healthz").status_code == 200, "health checks carry no credentials"
        ok = client.post(
            "/api/search",
            files={"file": ("q.jpg", b"fake", "image/jpeg")},
            data={"top_k": "5"},
            headers={"x-api-key": "secret"},
        )
        assert ok.status_code == 200


class TestUploadLimitConstant:
    def test_limit_is_a_sane_size_for_one_photograph(self):
        assert 1 * 1024 * 1024 <= MAX_UPLOAD_BYTES <= 32 * 1024 * 1024
