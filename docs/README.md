# Documentation index

Everything a reviewer needs, ordered from "what is this" to "show me the evidence".

## Read first

| Document | What it answers |
|---|---|
| [`../README.md`](../README.md) | What the system does, how to install and run it, and the two v1 corrections. |
| [`BENCHMARK.md`](BENCHMARK.md) | The measured comparison: baselines vs. the upgraded system, protocol, and significance. |
| [`RESEARCH-RETRIEVAL-LIMITS.md`](RESEARCH-RETRIEVAL-LIMITS.md) | How much retrieval quality is left inside the descriptor (proved with four methods) and what the remaining 16 errors actually consist of. |
| [`WEB.md`](WEB.md) | The search UI and JSON API: startup, measured latency, upload/readiness contract, deployment notes. |
| [`HISTORY.md`](archive/HISTORY.md) | Where the v1 scripts went, which claims were wrong and why, and every defect found in v2. |
| [`HYGIENE.md`](HYGIENE.md) | The pre-commit guards, why they exist, and why they carry no exception lists. |
| [`PASS-CI-FIRST-TRY.md`](PASS-CI-FIRST-TRY.md) | The pre-push checklist, the three ways a local check can pass while CI fails, and the exit-code contract. |

## Evidence

These are the raw outputs behind the claims, kept because a summary is not evidence.

| Document | Content |
|---|---|
| [`diagnostics/CALFW-IS-NOT-CAT-FACES.md`](diagnostics/CALFW-IS-NOT-CAT-FACES.md) | Full audit proving a dataset labelled "Calfw" contains human faces, with the positive control that validates the audit tooling. |
| [`diagnostics/calfw-species-audit.json`](diagnostics/calfw-species-audit.json) | YuNet human-face detection rate on the CALFW sample. |
| [`diagnostics/calfw-content-audit.json`](diagnostics/calfw-content-audit.json) | ImageNet-observed class distribution and cat-class probability mass. |
| [`diagnostics/avito-dinov2s-calfw.json`](diagnostics/avito-dinov2s-calfw.json) | A domain-trained animal-identification model scoring at chance on the same pairs. |
| [`diagnostics/oiid-reference-species-audit.json`](diagnostics/oiid-reference-species-audit.json) | Positive control: the same audit applied to images known to be cats. |
| [`diagnostics/cat-individuals-analysis.json`](diagnostics/cat-individuals-analysis.json) | Scale, resolution, quality and species composition of the evaluation corpus. |
| [`diagnostics/zeroshot-results.json`](diagnostics/zeroshot-results.json) | Untuned backbone comparison, retained for reference. |
| [`diagnostics/species-recognition.json`](diagnostics/species-recognition.json) | All-species statistics: cross-species separability before and after identity training. |
| [`diagnostics/final-untuned.json`](diagnostics/final-untuned.json) | The main benchmark table, machine-readable. |
| [`diagnostics/postprocess-tuning.json`](diagnostics/postprocess-tuning.json) | All 35 post-processing candidates with their val scores. |
| [`diagnostics/error-decomposition.json`](diagnostics/error-decomposition.json) | Per-query rank of the correct match and margins: the evidence that every remaining failure is a near tie. |
| [`diagnostics/research-round1.json`](diagnostics/research-round1.json) | k-reciprocal and diffusion scored against the baseline on identical descriptors. |
| [`diagnostics/identity-aggregation.json`](diagnostics/identity-aggregation.json) | The full k = 1..30 sweep showing identity pooling peaks at k = 2 and degrades monotonically after. |
| [`diagnostics/multiscale-fusion.json`](diagnostics/multiscale-fusion.json) | 224 px and 280 px descriptors alone and fused, with the mINP gain that is the one significant improvement. |
| [`data-findings.json`](data-findings.json) | Machine-readable record of what each dataset actually labels. |

## Archive

Superseded or third-party material, kept so nothing is silently discarded. See
[`archive/HISTORY.md`](archive/HISTORY.md) for the disposition of every file.

| Path | Content |
|---|---|
| `docs/archive/2026-10-catface-v1/scripts/` | The original numbered pipeline, verbatim. |
| `archive/2026-10-catface-v1/` | v1 setup note. |
| `archive/2026-10-calfw-audit/` | Derived imagery used to establish the CALFW finding. |
| `archive/third-party/` | A third-party paper's extracted text and figures, for local reading. Not distributed (see `.gitignore`). |

## Images

`images/` holds the search-result demonstrations referenced by the README.
