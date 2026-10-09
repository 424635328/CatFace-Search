# Container image for the search service.
#
# Multi-stage on purpose: the build stage needs a compiler and the full dependency graph, while the
# runtime stage needs neither. The final image therefore carries no build toolchain, which is both a
# size decision and an attack-surface decision.
#
# The model weights and the gallery manifest are NOT baked in. They are 86 MB and 25 MB of
# artifacts that change independently of the code, and a deployment that rebuilds an image to swap
# a checkpoint is a deployment that cannot roll back a model without also rolling back the service.
# Mount them instead; the entry point fails fast with an explanatory message when they are absent.

# ------------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------------
FROM python:3.11-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# CPU-only torch: the default wheel pulls several gigabytes of CUDA libraries that this image does
# not use. A GPU deployment should start from an nvidia/cuda base instead and mount the same code.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy only what the build needs, so a change to docs/ or tests/ does not invalidate the dependency
# layer and force a full reinstall on every commit.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[web]"

# ------------------------------------------------------------------------------------------------
# runtime
# ------------------------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# An unprivileged user by default: a container that can write to its own code is a container that
# can be modified by whatever it is exploited with.
RUN groupadd --system --gid 1001 catface \
 && useradd --system --uid 1001 --gid catface --create-home catface

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CATFACE_HOST=0.0.0.0 \
    CATFACE_PORT=8000 \
    CATFACE_DEVICE=cpu \
    CATFACE_CHECKPOINT=/data/best.pt \
    CATFACE_MANIFEST=/data/cat_individuals_manifest.jsonl

COPY --from=build /opt/venv /opt/venv
WORKDIR /app

USER catface

EXPOSE 8000

# /healthz answers without touching the model, so it is the correct probe for liveness. Readiness is
# /api/status; an orchestrator that conflates the two restarts a healthy process while the model is
# still loading.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

ENTRYPOINT ["python", "-m", "catface.web"]
