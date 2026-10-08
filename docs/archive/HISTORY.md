# Project history and file disposition

This repository was restructured from a set of numbered scripts into a package
(`src/catface/`) with a test suite, a benchmark harness and reproducible data acquisition.
Nothing was silently discarded: every superseded file is either kept in `archive/` or
accounted for below, with the reason it is no longer active.

---

## 1. Where the old entry points went

| v1 file | Replaced by | Why it was superseded |
|---|---|---|
| `01_crop_faces.py` | `catface prepare --source <corpus>` → `src/catface/data/prepare.py`, `data/cropping.py` | Cropped the top third of a whole-body YOLOv9 box as a stand-in for a face, with no way to tell a good crop from a bad one. Now uses annotated face boxes where available, records per-crop quality signals (sharpness, luminance, contrast) and content-hashes each tile for duplicate detection. |
| `02_generate_embeddings.py` | `catface prepare` + `catface train` + `catface index` | Both generated embeddings *and* owned the model definition, so training and inference could drift apart. Now one `Embedder` serves both, and embeddings are produced by an index build rather than a standalone script. |
| `03_search_similar.py` | `catface index --query` → `src/catface/index/faiss_index.py` | Was a demo: the query path was a hardcoded string and results went to a Matplotlib window. Now a reusable `VectorIndex` returning ids and scores, with a CLI surface. |
| `04_add_new_cat.py` | `catface index` (rebuild or extend) | Duplicated the dataset/model/transform pipeline of `02`, so a change to preprocessing had to be made in two places. |
| `clean_database.py` | `catface index` rebuild | Cleaned up dangling file paths in a pickle. A rebuild from the manifest is simpler and cannot desynchronise ids from vectors. |
| `download_model.py` | `catface acquire` → `src/catface/data/sources.py` | Downloaded one hardcoded checkpoint with no integrity check. Now a catalogue of datasets with expected byte counts, resumable transfer and post-download verification. |
| `cmd.txt` | see `docs/archive/2026-10-catface-v1/cmd.txt` | A note to manually downgrade NumPy to 1.24.6 because v1's dependency set pulled 2.x. The v2 stack is consistent (NumPy 2.4.6 with torch 2.6.0) so the workaround is obsolete. |

All seven are preserved verbatim in `docs/archive/2026-10-catface-v1/` (the scripts
themselves are in `docs/archive/2026-10-catface-v1/scripts/`) so that a reader can check what the previous
behaviour actually was rather than taking this table on trust.

---

## 2. Corrections to v1's claims

Two statements in the original README were not supported by the data, and both invalidated
conclusions that had been drawn from it. They are documented in full rather than quietly
dropped.

### 2.1 Oxford-IIIT Pet has breed labels, not individual identity

The v1 README's premise was that per-cat identity was available from this corpus. It is not:
the file-name prefix is the **breed** (`Abyssinian`, `Bengal`, …; 12 values for cats, ~99
images each) and `list.txt` documents its own ID column as `CLASS-ID` over 37 classes.

Consequence: the corpus cannot train or evaluate *individual* cat identification. It is a
breed classifier in disguise. Evidence: `docs/data-findings.json`.

The corpus is still used, but only for what it does provide — a labelled, head-box-annotated
population used as a negative control for the species audits.

### 2.2 CALFW on HuggingFace is a human-face dataset

The `calfw` split of `cat-claws/face-verification` (6000 pairs, 112×112) was adopted as the
verification benchmark and produced AUC 0.59–0.64 for every model. That number was
implausible enough to audit, and the audit was conclusive:

| Observer | Known cat images (control) | CALFW |
|---|---|---|
| OpenCV YuNet **human**-face detector | 10/150 (6.7 %), mean conf 0.054 | **300/300 (100 %)**, mean conf 0.920 |
| ImageNet ResNet-50 V2, cat-class mass | — | median 0.0027; 0/600 images predicted as a cat |
| Domain-trained animal-ID model (695 091 pets) | — | ROC-AUC 0.5613 |

The images are human faces, most likely LFW rescaled to 112×112. The whole audit is in
`docs/diagnostics/CALFW-IS-NOT-CAT-FACES.md`, with the raw observer outputs archived in
`docs/archive/2026-10-calfw-audit/`. The protocol was withdrawn and replaced by the
individual-identity corpus described in `docs/data-findings.json`.

The v1 pipeline itself did not cause this; the *validation* of it did, by trusting a dataset
name and card over the pixels.

---

## 3. Defects found and fixed during the v2 work

Recorded because each one produced a plausible-looking but false result, so a future reader
benefits from knowing the symptom.

| Symptom | Root cause | Fix |
|---|---|---|
| Descriptor shape wrong for any input size other than 224×224 | v1 built the backbone with `nn.Sequential(*list(resnet50(children))[:-1])`, which also dropped the final `AdaptiveAvgPool2d`. At 224×224 the feature map is already 7×7, so the numbers looked right. | `TorchvisionBackbone` keeps the pooling layer and applies it explicitly. |
| Training loss constant at 30.45 for ten epochs with the model visibly not learning | The positional-embedding swap for non-native resolutions was restored in a `finally` block *before* the backward pass. Gradient checkpointing recomputes the forward during backward, so the recomputation disagreed with the recorded graph; the AMP scaler treated the resulting overflow as a bad step and discarded the update. | The swap is now a context manager (`resolution_scope`) that must stay open across forward *and* backward, and a checkpointed backward outside it raises instead of failing silently. |
| Same symptom, second cause | fp16 autocast produced `inf` gradient norms on many steps, making `clip_grad_norm_` return NaN and discarding those updates. | Training uses bfloat16, which shares fp32's exponent range and needs no loss scaling. Measured: fp16 → frequent `inf`; bf16 → none. |
| Convergence pathologically slow | `grad_clip` was 1.0 while measured gradient norms were 200–900, so ~99 % of every update was clipped away. | Default raised to 10.0, measured rather than inherited. |
| Checkpoint could not be reloaded after a non-native-resolution forward | The positional embedding was replaced in place, so the *registered* parameter's shape depended on the last resolution used. | The native embedding is kept intact and a cached resampled tensor is swapped in only for the duration of a forward, making interpolation idempotent and `state_dict` shapes stable. |
| Report showed `R@1 = 0.0517` next to a confidence interval of `[0.86, 0.92]` | The column was labelled `R@k` but held *full recall*, while the CI in the same table was computed on the *top-k hit rate*. | Metrics renamed to state the quantity (`hit@k`, `fullR@k`, `AP@k`), and reported separately. |
| `descriptor_dim` read 512 for every backbone including ResNet-50 | It reported the head's projected width, not the backbone's pooled width. | Reports `backbone_dim`, `projected_dim` and the compared `descriptor_dim` separately. |
| `'MetricHead' object is not callable`, `'Embedder' object is not callable` | Both are plain classes owning an `nn.Module` rather than being one, so `forward` was never wired to `__call__`. | Explicit `__call__ = forward`. |
| `Expected more than 1 value per channel` when embedding a single image | `BatchNorm1d` in the head was in train mode. | `Embedder.embed` is a pure inference path: it forces eval mode, disables autograd, and restores the previous mode. |
| Batch labels could disagree with the images they described | `Trainer` built the dataset from unfiltered records while `PKBatchSampler` filtered out single-image identities internally, so the sampler's indices addressed the wrong entries. | Both use one filtered, shared index space, asserted at construction. |
| `mINP` disagreed with the published definition | The last-relevant-rank computation reversed the mask and read the resulting index directly. | Corrected and pinned by a hand-computed test. |
| Recall was understated whenever the query's own image was in the gallery | The self-match was excluded from *relevance* but left in the *ranking*, so it occupied a top-k slot. | Excluded pairs are removed from the ranking (`-inf`), which is the production situation since a query is usually already indexed. |

---

## 4. What is archived, and what is not recoverable

**Archived and readable:** v1 scripts, the v1 setup note, the CALFW audit evidence and its
derived imagery, a third-party paper's extracted text and figures (kept locally only; see
`.gitignore` for why they are not committed).

**Deliberately deleted:** `embeddings_tta.pkl.bak` (19 MB). It was a backup of a v1 embedding
database whose images the repository does not contain, so it could not be validated or used;
it is regenerable from the manifest by `catface index`. No information was lost, because the
file was a stale copy of a derived artifact rather than a source.

**Not recoverable, and worth stating:** the v1 system's *measured accuracy on a cat-face
task* was never actually established. The corpus it was benchmarked against is the human-face
dataset described in §2.2, and the corpus it was trained on had no individual labels per
§2.1. Any v1 performance figure that circulated should be treated as unverified.
