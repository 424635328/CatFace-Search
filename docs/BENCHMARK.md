# Benchmark: what was measured, and against what

Every number here is reproducible: the command that produced each table is given with it, and
the raw run output is kept in `docs/diagnostics/` and `artifacts/benchmarks/`.

---

## 1. Protocol

| Field | Value |
|---|---|
| Corpus | Kaggle *Cat Individual Images* — 13 106 photos of 509 individual cats (CC BY 4.0) |
| Preparation | 12 644 usable 256×256 face tiles after quality filtering; 462 duplicate tiles removed |
| Split policy | **Identity-disjoint**: 352 train / 75 val / 76 test individuals, asserted for leakage at load time |
| Training | 8 833 images of 352 identities |
| Query/gallery | 1 query per test identity; remaining 12 141 images form the gallery |
| Self-match | Removed from the ranking, not merely marked irrelevant |
| TTA | Horizontal flip averaged with the identity view |
| Input size | 224×224 |
| Selection | Post-processing hyper-parameters are chosen on **val** identities and reported on **test** |
| Seed | 1337 (`split_sha1 = f78330dcdffd1adda1664f5cecf0c78454d0d454`) |

Why identity-disjoint matters: if two photographs of the same cat fall on opposite sides of
the split, the score measures memorisation. The split is re-asserted at training time
(`assert_identity_disjoint`), not merely assumed.

Two gallery sizes appear below and are labelled wherever they do: **full** (503 query
identities, 12 141 gallery images) and **compact** (75 val / 76 test identities, 1 834
gallery images). The compact protocol is what the post-processing ablation uses, because the
transformations are fitted on the gallery and that must be done without touching test.

> The two protocols are **not** comparable. The compact one comes from
> `data/manifests/cat_individuals_splits/test.txt` and is also what the training loop prints per
> epoch as its held-out check; the full one takes one query per identity from the *whole*
> manifest. Measured compact `hit@1` for the trained DINOv2-S is **0.9868**, higher than the
> published full-protocol **0.9682** purely because the gallery is 6.6× smaller. Quoting the
> compact number as the benchmark result would overstate it by 1.9 points.

## 2. Metric definitions

These are reported separately because they answer different questions, and conflating them
is the most common way a retrieval benchmark is overstated.

| Metric | Question it answers |
|---|---|
| `hit@k` | For how many queries is **at least one** same-identity image in the top k? (CMC / top-k accuracy — the standard re-identification number.) |
| `fullR@k` | What share of that identity's gallery images appear in the top k? |
| `AP@k` | Ranking quality averaged over the relevant items inside the top k. |
| `mINP` | Recall normalised by the rank of the **last** relevant item; does not saturate once k covers every relevant item. |
| `mRR` | Mean reciprocal rank of the first correct hit. |

`hit@k` and `fullR@k` are *not* interchangeable. On this corpus each identity has ~24 gallery
images, so retrieving one correct image gives `hit@1 = 1.0` but `fullR@1 ≈ 0.04`. A report
quoting the second while calling it recall@1 understates by 25×; a report quoting the first
while calling it recall overstates. Both are therefore printed.

## 3. Main result — retrieval on unseen identities

Full protocol (503 query identities, 12 141 gallery images), untuned descriptors. Raw output:
`docs/diagnostics/final-untuned.json`.

| Configuration | params | hit@1 | hit@5 | hit@10 | mINP | mRR |
|---|---|---|---|---|---|---|
| baseline-resnet50 — *the v1 backbone* | 25.6 M | 0.8867 | 0.9364 | 0.9483 | 0.2882 | 0.9089 |
| dinov2-small, zero-shot | 21 M | 0.9344 | 0.9742 | 0.9801 | 0.3945 | 0.9515 |
| dinov2-base, zero-shot | 86 M | 0.9523 | 0.9801 | 0.9881 | 0.4352 | 0.9645 |
| **dinov2-small + ArcFace, trained** | **21 M** | **0.9682** | **0.9881** | **0.9901** | **0.7686** | **0.9764** |
| dinov2-base + ArcFace, trained | 86 M | 0.9583 | 0.9861 | 0.9901 | 0.7219 | 0.9700 |

hit@1 with 95 % bootstrap CI: ResNet-50 `[0.8588, 0.9105]`, DINOv2-S zero-shot
`[0.9145, 0.9543]`, DINOv2-B zero-shot `[0.9324, 0.9722]`, trained DINOv2-S `[0.9523, 0.9821]`,
trained DINOv2-B `[0.9404, 0.9752]`. The two trained models' intervals overlap, so the 1.0-point
gap between them is **not** established as significant by this test alone.

### 3.1 这份语料的标注缺陷，以及它对 hit@1 的影响

上表的数字按**标签**计分。该语料有 137 对"**同一张照片挂在两个身份标签下**"（5 组标签：
`0455`/`0455_357`、`0077`/`0083`、`0046`/`0082`、`0314`/`0317`、`0335`/`0337`；`0455_357` 其实是
`0455` 的子目录）。后果是 503 个查询里有 **5 个被判错，而它检索到的 top-1 就是自己的照片**——
相似度 0.9994–0.9999，属于标注缺陷而非模型错误。

| 口径 | hit@1 |
|---|---|
| 标签口径（上表，可直接与其它工作对比）| 0.9682 |
| 剔除 5 个标注缺陷 | **0.9781** |

**两个数都要报。** 只报 0.9781 等于把数据缺陷算成自己的成绩；只报 0.9682 而不说明，
读者会以为 16 个错误全是模型的。完整证据、像素级判据与成因见
[`docs/diagnostics/LABEL-COLLISIONS.md`](diagnostics/LABEL-COLLISIONS.md)。

> 另注：按标签做身份不相交划分**无法**发现这类问题——`assert_identity_disjoint` 比较的是标签，
> 不是照片。换语料时应先跑 `python -m tools.find_duplicates`。

### 3.2 识别最可靠的那一张图

`hit@1 = 0.9682` 是 503 个查询的**均值**，看不出分布，也回答不了"哪张图识别得最好"。
`python -m tools.find_best_match` 把每个查询单独打分并排名（`docs/diagnostics/best-match-dinov2s.json`）。
"最可靠"有三个不同的定义，答案不同，所以三个都报：

| 口径 | 含义 | 结果 |
|---|---|---|
| **margin**（默认）| top-1 正确匹配 减去 最强错误身份 的差距 | 见下 |
| sim | 单纯相似度最高（**可能是自信的错误**）| 见下 |
| worst | 排名最末 | 见下 |

**最可靠（margin 最大）**——识别毫无歧义：

| | 值 |
|---|---|
| 查询 | `data/faces/cat_individuals__0490_0490_004.jpg.jpg`（身份 `0490`）|
| top-1 匹配 | `cat_individuals__0490_0490_006.jpg.jpg` |
| 相似度 | 0.9757 |
| **margin** | **0.6308**（最强错误身份只有 0.3448）|
| 该身份在图库中的照片数 | 18 |

**最自信（sim 最高）——但它是个陷阱**：`0077_031` 与 `0083_024` 相似度 0.9999992，
判定**错误**。原因就是 3.1 节的标注缺陷：这是同一张照片换了个标签。
**"相似度最高"不等于"识别最准"**，这正是本工具把 margin 作为默认口径的原因。

**最不可靠（margin 最小）**：`0335_002`（身份 `0335`）匹配到 `0337_016`，
相似度 0.9996、margin **−0.4568**——同样是标注碰撞（`0335`/`0337`），
错误身份几乎满分，而正确身份只有 0.5428。**失败案例一并列出，否则读者会以为每张图都像第一名那样。**

Paired bootstrap against the baseline — the same queries are resampled for both models, which
cancels the shared task difficulty and detects differences that independent resampling misses:

| Candidate | Δ hit@1 | 95 % CI | P(not better) |
|---|---|---|---|
| dinov2-small, zero-shot | +0.0477 | [+0.0278, +0.0676] | 0.000 |
| **dinov2-small + ArcFace, trained** | **+0.0815** | **[+0.0596, +0.1044]** | **0.000** |
| dinov2-base + ArcFace, trained | +0.0716 | [+0.0497, +0.0954] | 0.000 |

### What this shows, and what it does not

**Shown.** Replacing the ImageNet-supervised ResNet-50 with a self-supervised DINOv2 backbone
gains 5.8 points of top-1 identification with no training at all (+4.6 for the small variant).
Adding identity-supervised metric learning on top takes it to +7.4 points, and improves `mINP`
by 2.7× — meaning the correct match is not merely present in the list but ranked near the top.
All three improvements are significant at p < 0.001 under a paired test.

**The most useful finding is that bigger is not better here.** The trained DINOv2-S
(21 M parameters) beats *both* the untrained DINOv2-B (86 M) by 1.6 points **and the trained
DINOv2-B (86 M) by 1.0 point**, while embedding 2.2× faster (121 s vs 271 s over 12 644
images) and training in 34 minutes instead of 47. Two things follow:

1. The gain comes from aligning the training objective with the retrieval task — a
   self-supervised encoder still optimises for "what is this", whereas ArcFace over identities
   optimises for "which one is this" — not from model capacity.
2. On a 352-identity training set, the larger backbone has *more* capacity than the task can
   use, and the extra capacity is not turned into better identity separation. This is the
   expected regime for small-corpus metric learning, and it is why the recommendation below is
   the small model.

The two trained models' confidence intervals overlap, so the 1.0-point gap should be read as
"the small model is at least as good and much cheaper", not as a demonstrated superiority of
the small model.

**Recommendation.** Ship `dinov2-small + ArcFace`: it is the best `hit@1` and `mINP` measured,
it is 4× smaller than DINOv2-B, it embeds 2.2× faster, and it trains in 34 minutes on a 6 GB
card. DINOv2-B is retained as a measured alternative, not as the default.

**Not shown.** These numbers are within one corpus, split by identity. They do not establish
generalisation to a different camera, species, or collection. Cross-corpus evaluation is not
reported because the two corpora available here share no identity labels; `docs/archive/HISTORY.md`
§2 explains why the originally intended cross-dataset benchmark could not be built.

## 4. Post-processing ablation

PCA-whitening, database-side augmentation (DBA) and αQE are label-free transforms fitted on
the gallery only. They are selected on val identities and reported on test. Raw output:
`docs/diagnostics/postprocess-tuning.json`.

```bash
python -m tools.tune_postprocess --checkpoint artifacts/train/dinov2s-arcface/best.pt \
  --out docs/diagnostics/postprocess-tuning.json
```

35 candidates evaluated on val; one selected and re-evaluated on test.

| Metric | untuned | selected (`pca` + DBA + αQE) | Δ |
|---|---|---|---|
| hit@1 | 0.9868 | **1.0000** | +0.0132 |
| hit@5 | 1.0000 | 1.0000 | 0.0000 |
| mINP | 0.6574 | **0.7234** | **+0.0661** |
| mRR | 0.9934 | **1.0000** | +0.0066 |

Compact protocol (76 test identities, 1 834 gallery images), so the absolute values are not
comparable to §3 — only the delta is.

**A methodological finding, recorded because it changed the answer.** On val, **19 of the 35
candidates tie on `hit@1` at 1.000** — the model has saturated that metric. A selection rule
that breaks such ties by grid order picks an arbitrary candidate, and the first version of this
script did exactly that: it selected "no post-processing" and reported a zero delta everywhere,
which would have been read as "post-processing does not help". Breaking the tie on `mINP` —
which still varies while `hit@1` is saturated — selects `pca + DBA + αQE` and produces the
+6.6 point `mINP` gain above.

So the honest statement is: **when the primary metric saturates, the tie-break decides the
result and must be chosen deliberately.** The tool now reports how many candidates tied, so a
saturated comparison is visible rather than silent.

## 5. All-species recognition statistics

The identity benchmark above asks "is this the same cat?". This section asks a different
question: **does the descriptor still organise when the population contains more than one
species?** It matters because the architecture is meant to be a general retrieval backbone, and
because a metric-learning run is capable of destroying general structure in exchange for
identity sensitivity. That is a measurable trade-off, so it is measured.

```bash
python -m tools.analyze_species_recognition \
  --checkpoints trained-dinov2s=artifacts/train/dinov2s-arcface/best.pt \
  --include-untrained --backbone dinov2_vits14 \
  --limit-per-species 800 --out docs/diagnostics/species-recognition.json
```

Corpus: Oxford-IIIT Pet, 1 600 images balanced 800 cat / 800 dog over 8 breeds. The
untrained-head row uses the *same* pretrained backbone with a freshly initialised head, which
separates "the backbone already did this" from "training preserved it".

| Model | species 1-NN | species linear probe | separation gap | breed 1-NN |
|---|---|---|---|---|
| trained dinov2-small + ArcFace | 0.9694 | 0.9825 ± 0.0064 | 0.0989 | 0.7881 |
| untrained head, same backbone | **1.0000** | **0.9994 ± 0.0013** | **0.2743** | **0.9506** |

`separation gap` = mean intra-species cosine similarity − mean inter-species similarity.

### The finding, stated plainly

**Training bought individual identity at the cost of general visual organisation.** The
species-separation gap fell by **64 %** (0.2743 → 0.0989) and breed 1-NN accuracy fell from
0.9506 to 0.7881. Concretely:

| Quantity | untrained head | trained | reading |
|---|---|---|---|
| mean similarity, same species | 0.3357 | 0.1986 | the space was pulled **inwards** |
| mean similarity, different species | 0.0614 | 0.0998 | the two species were pushed **together** |
| descriptor variance | 0.001565 | 0.001661 | **not** collapsed |

This is not a collapsed descriptor — variance is unchanged — it is a *compressed* one: the
margin between "same cat" and "different cat" was bought by shrinking the margin between "cat"
and "dog". Species is still decodable (linear probe 0.9825, chance 0.5), so the capability is
degraded rather than erased, but the architecture **cannot** be reused as a species or breed
classifier after identity training without either a multi-task objective or a second head.

Two implications worth acting on:

1. **Do not reuse an identity-trained embedder for species filtering.** If a deployment needs
   "is this even a cat?", use a separate classifier. The trained descriptor will still find the
   right cat when one is present, but it is measurably worse at saying whether there is one.
2. **If both capabilities are needed, train them together.** The trade-off is an artefact of a
   single-objective loss, not a property of the backbone. Adding a species-classification term
   (or a distillation term against the frozen backbone) is the standard remedy; measuring the
   gap as above is how its cost would be verified.

## 6. What is reported and what is deliberately absent

| Planned comparison | Status |
|---|---|
| Backbone comparison (ResNet-50 → DINOv2) | Reported, §3 |
| Identity-supervised metric learning | Reported, §3 |
| Retrieval post-processing ablation | Reported, §4 |
| All-species recognition statistics | Reported, §5 |
| Verification metrics (AUC / EER / TAR@FAR) | **Withdrawn, not measured.** The intended pair corpus turned out to be human faces, not cats — see `docs/diagnostics/CALFW-IS-NOT-CAT-FACES.md`. The harness supports it (`catface verify`, `tools/run_verification`); it needs a genuine cat pair set. No substitute corpus was available, so this capability remains **unverified** rather than verified-bad. |
| Cross-dataset generalisation | **Not possible with the corpora used.** The two available corpora share no identity labels, so no positive cross-corpus pair exists and no honest number can be produced. |
| Breed-level identity retrieval | **Withdrawn.** Oxford-IIIT Pet labels breeds (12 cat values), not individuals, so it cannot measure identity retrieval. §5 uses it for species/breed *organisation*, which is what it actually labels. |
| Detector fine-tuning (the v1 YOLOv9 cat-face detector) | **Not done, and not currently needed.** The trained pipeline crops from annotated head boxes where the corpus provides them (Oxford) and records `detector='whole_image'` where it does not (Cat Individual Images). On the latter, whole-image framing already reaches hit@1 = 0.9682 (0.9781 excluding the 5 label defects of §3.1), so a detector is a possible accuracy gain rather than a blocking gap. |

Stating the absences explicitly is part of the result: an unmeasured capability that is silently
omitted reads as a passing one. The verification row is the one that matters most — it is the
only place where a user-facing confidence threshold would depend on a number nobody has measured.

## 7. Training recipe notes that affect the numbers

These were established by measurement, not inherited, and each one changed the result:

| Setting | Value | Why |
|---|---|---|
| Precision | bfloat16 | fp16 produced `inf` gradient norms on many steps, which made `clip_grad_norm_` return NaN and silently discarded those optimiser updates. bf16 shares fp32's exponent range and needs no loss scaling. |
| `grad_clip` | 10.0 | Measured gradient norms on this corpus are 200–900. A clip of 1.0 discarded roughly 99 % of every update and was the dominant reason an earlier run did not converge. |
| Batch composition | 16 identities × 4 images | PK sampling. With a random shuffle, a batch rarely holds two images of the same cat, so the margin loss sees almost no positive pairs. |
| Optimiser | AdamW, `lr` 1e-3 with backbone scaled ×0.1 | The head is randomly initialised while the backbone is pretrained; equal learning rates destroy the pretrained features within a few hundred steps. |
| Gradient checkpointing | off (batch 64 fits in 6 GB) | Halves the per-step cost when memory allows. It must stay on for DINOv2-B, whose activations do not fit otherwise. |
| Model selection | best held-out identity `hit@1` | Validation loss is often anti-correlated with retrieval quality, so it is not used for selection. |
| Epochs | 15 of 20 (early-stopped) | Best epoch 7, then eight epochs without improvement. Loss fell monotonically from 24.0 to 1.51. |

### 7.1 Pause and resume

A run checkpoints after **every** epoch, so it can be stopped and continued without losing
work or changing the optimisation trajectory. Resume is faithful because the state carries the
optimiser moments, the scheduler position, the sampler's epoch and the RNG states — not just
the weights. `tests/test_resume.py` asserts that equivalence rather than merely that a resumed
run executes.

```bash
# Start with a wall-clock budget; it pauses cleanly at an epoch boundary.
python -m tools.train_embedder --backbone dinov2_vits14 --epochs 20 \
  --output artifacts/train/dinov2s-arcface --max-seconds 3600

# Continue where it stopped (same command plus --resume).
python -m tools.train_embedder --backbone dinov2_vits14 --epochs 20 \
  --output artifacts/train/dinov2s-arcface --resume
```

| Mechanism | How to trigger |
|---|---|
| Wall-clock budget | `--max-seconds <s>` |
| Signal | `Ctrl-C` (SIGINT) or SIGTERM — pauses at the next epoch boundary instead of aborting mid-epoch |
| External request | create `<output>/PAUSE`; the run stops at the next epoch boundary |
| Resume | `--resume` (needs the same `--epochs` or a larger value) |

Files written into the run directory:

| File | Purpose |
|---|---|
| `train_state.pt` | Resumable state: weights, optimiser, scheduler, RNG, history. Written atomically every epoch. |
| `best.pt` | Best-validation weights, for inference (index build, benchmark). Also written on pause. |
| `training_history.json` | Per-epoch record plus `pause_reason`, so a paused run is never reported as converged. |
| `run.json` | Recipe, environment, and the held-out metric re-measured from the saved artifact. |

A paused run is deliberately **not** flagged as early-stopped: `pause_reason` is set instead,
so an interrupted experiment cannot be mistaken for a finished one. On pause the *latest*
weights are kept (not the best epoch's), because resuming from the best epoch would follow a
different trajectory than an uninterrupted run — the reproducibility this feature exists to
protect.

### 7.2 Measured cost

Reference machine: RTX 3060 Laptop 6 GB, DINOv2-S, 8 833 training images, batch 64, no
gradient checkpointing.

| Quantity | Measured |
|---|---|
| Per epoch | **239 s** (≈120 s for the 138 training batches, ≈90 s validating against 1 901 images) |
| 20-epoch run | 34 min (early-stopped after 15) |
| Descriptor extraction, 12 644 images | 141 s (DINOv2-S), 326 s (DINOv2-B), 116 s (ResNet-50) |
| Post-processing grid, 35 candidates | 8 s, after embedding both protocols once |

Validation is not free and is about 38 % of per-epoch cost, so it should be counted when sizing
a run. Embedding dominates everything else, which is why the tuning grid embeds once and then
reuses the vectors.

## 8. Environment

```
python 3.11.16 | torch 2.6.0+cu124 | torchvision 0.21.0+cu124 | faiss 1.15.1
NVIDIA GeForce RTX 3060 Laptop GPU (6 GB, compute capability 8.6)
```

Recorded because a comparison between two numbers is only meaningful if they came from
comparable hardware and library versions. `artifacts/*/metrics.json` embeds the same fingerprint
alongside every run.
