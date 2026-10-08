# CatFace Search — 猫脸身份检索系统

<p align="center">
  <img src="docs/images/search-result-01.png" width="80%" alt="效果展示"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![CI](https://github.com/424635328/CatFace-Search/actions/workflows/ci.yml/badge.svg)

一个"以图搜猫"引擎：给一张猫脸照片，在库里找出**同一只猫**的其他照片。

核心设计目标是**可扩展性**——加入新猫不需要重新训练，只需增量更新索引。

---

## 📊 实测效果

在 **503 个未参与训练**的猫个体、12 141 张画廊图的检索协议上（完整报告见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)）：

| 配置 | 参数量 | hit@1 | hit@5 | mINP | 嵌入耗时 |
|---|---|---|---|---|---|
| 旧版骨干 ResNet-50（ImageNet 预训练） | 25.6 M | 0.8867 | 0.9364 | 0.2882 | 95 s |
| DINOv2-S 零样本 | 21 M | 0.9344 | 0.9742 | 0.3945 | 121 s |
| DINOv2-B 零样本 | 86 M | 0.9523 | 0.9801 | 0.4352 | 326 s |
| **DINOv2-S + ArcFace 微调** | **21 M** | **0.9682** | **0.9881** | **0.7686** | **121 s** |
| DINOv2-B + ArcFace 微调 | 86 M | 0.9583 | 0.9861 | 0.7219 | 271 s |

配对 bootstrap 显著性检验：微调 DINOv2-S 相对 ResNet-50 **+8.15 个点**（95% CI [+5.96, +10.44]，p<0.001）。嵌入耗时是 12 644 张图的总时长。

三点值得注意：

1. **21 M 的微调 DINOv2-S 比 4 倍大的 86 M 微调 DINOv2-B 还高 1.0 个点**，且嵌入快 2.2 倍、训练少 13 分钟。收益来自让训练目标与检索任务对齐（自监督优化"这是什么"，ArcFace 优化"这是哪一只"），**不是**来自模型容量——在 352 个身份的训练集上，大骨干的额外容量没被用上。
2. **mINP 提升 2.7 倍**，说明正确的匹配不只是"出现在列表里"，而是排得足够靠前。
3. **代价已量化**：身份训练会压缩通用视觉组织能力——物种可分间隔下降 64%，品种 1-NN 从 0.9506 降到 0.7881。详见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md) §5。

---

## ⚠️ v2 重要更正

本次升级修掉了旧版中的两处**实质性错误**，它们让项目此前的结论不成立：

| 问题 | 旧版 | v2 |
|---|---|---|
| **Oxford-IIIT Pet 被当作身份数据集** | README 声称可从中获得身份信息 | 已核实：其文件名前缀是**品种**（12 个），`list.txt` 是 `CLASS-ID 1..37`。**该数据集只有品种标注，没有个体标注**，无法用于个体识别训练/评测。详见 [`docs/data-findings.json`](docs/data-findings.json) |
| **CALFW 被当作猫脸验证基准** | 用作验证数据集 | 已核实：该 split 的 6000 张图**100% 是人脸**（YuNet 人脸检测器 300/300 命中，均值置信度 0.92）。详见 [`docs/diagnostics/CALFW-IS-NOT-CAT-FACES.md`](docs/diagnostics/CALFW-IS-NOT-CAT-FACES.md) |

两处都保留了完整审计证据链。**这些数据集的名称与 dataset card 不构成其内容的证据。**

---

## 🧠 原理

不训练分类器，而是把"识别"转化为"在高维空间测距离"：

1. **用度量学习而非分类** — ImageNet 分类器优化的是"类间线性可分"，对**类内方差**没有约束。同一只猫的不同照片在该空间里可以随意散开。身份检索需要的是"类内紧凑"，这是一个不同的目标函数。
2. **把脸变成指纹** — 骨干网络把每张猫脸映射为单位范数向量（embedding）。
3. **在指纹空间找最近邻** — FAISS 索引毫秒级返回最相似的脸。
4. **身份从几何中涌现** — 系统不知道"身份"概念，但同一只猫的向量天然聚在一起。

正因为不需要训练分类器，**加入新猫只需更新索引，无需重新训练**。

---

## 🏗️ 架构

```
cat_retrieval/                     data/（原始数据 + 各类派生数据）
├─ raw_images/   原始猫图           ├─ oxford_annotations/   OIID 标注（含头部框）
└─ cropped_faces/ 裁剪后的猫脸      ├─ raw/                  下载的原始语料
                                    ├─ faces/                标准化后的 256×256 脸块
                                    └─ manifests/            清单 + 身份隔离划分

src/catface/                       企业级包结构
├─ config.py          类型化配置（未知键报错、指纹可追溯）
├─ errors.py          异常层级
├─ logging_utils.py   结构化日志 + 环境指纹
├─ data/
│  ├─ sources.py      数据集目录、完整性校验下载、安全解压
│  ├─ annotation.py   标注解析（OIID XML、pair parquet）
│  ├─ cropping.py     裁剪几何 + 质量信号
│  ├─ manifest.py     清单 + **按身份**划分 + 泄漏检测
│  └─ prepare.py      各语料 → 脸块清单
├─ models/
│  ├─ backbone.py     骨干注册表（可扩展）
│  ├─ pooling.py      池化（auto/gap/gem/cls/cls_gap）
│  ├─ heads.py        ArcFace / CosFace / SubCenter / Triplet
│  └─ embedder.py     统一推理入口（训练与推理不漂移）
├─ train/loop.py      PK 采样 + 分层学习率 + EMA + 按检索指标选模型
├─ eval/
│  ├─ metrics.py      R@k / fullRecall@k / mAP / mINP / mRR / AUC / EER / TAR@FAR
│  ├─ postprocess.py  PCA白化 / DBA / αQE（免训练提点）
│  ├─ protocols.py    查询/画廊协议 + 身份泄漏断言
│  └─ benchmark.py    统一评测循环 + 配对 bootstrap 显著性检验
├─ index/faiss_index.py  FAISS 索引（带 NumPy 回退，两者结果一致）
└─ cli.py             catface doctor/acquire/prepare/train/benchmark/verify/index
```

---

## 🚀 快速开始

```bash
# 1. 环境（PyTorch + CUDA 请按官方指引安装）
python -m venv .venv && . .venv/bin/activate
pip install -e ".[benchmark,dev]"          # 或 ".[cpu]" 用于纯 CPU

# 2. 自检：设备、库版本、数据就绪度（不做任何修改）
catface doctor --config configs/default.yaml

# 3. 取数据（自动校验大小，支持断点续传 + 多流并行）
catface acquire --datasets oxford_iiit_pet

# 4. 裁剪猫脸 + 建立清单 + **按身份**划分
catface prepare --source oiid_cat

# 5. 训练度量学习嵌入
catface train --config configs/default.yaml --backbone dinov2_vitb14 --epochs 25

# 6. 基准对照（基线 vs 升级，含配对显著性检验）
python -m tools.run_benchmark_suite --protocol cat_individuals

# 7. 建索引 + 查询
catface index --checkpoint artifacts/train/best.pt \
              --manifest data/manifests/cat_individuals_manifest.jsonl \
              --query query/my_cat.jpg --top-k 10
```

---

## 📊 评测报告

每次 benchmark 产出 `artifacts/benchmarks/<run>/`：

- `summary.md` — 可直接贴进评审的对照表
- `metrics.json` — 完整指标 + **环境指纹**（commit、库版本、GPU）+ 协议描述 + split 内容哈希

报告包含：

| 指标族 | 指标 | 回答的问题 |
|---|---|---|
| 检索 | `R@k` | 前 k 张里有没有同一只猫？ |
| 检索 | `fullRecall@k` | 这只猫的所有照片，找回了多少比例？ |
| 检索 | `mAP@k`、`mINP`、`mRR` | 排序质量（mINP 在 mAP 饱和后仍能区分模型） |
| 验证 | `ROC-AUC`、`EER` | 阈值无关的整体判别质量 |
| 验证 | `TAR@FAR` | **在 1% 误接受预算下，漏掉了多少真匹配？** |
| 统计 | 配对 bootstrap | 提升是真的，还是抽样噪声？ |

> `R@k` 与 `fullRecall@k` 是两个不同的问题（前者是命中率 CMC，后者是召回比例），混淆二者是检索评测最常见的夸大来源，因此分开报告。
> 配对 bootstrap 用**相同查询**重采样，抵消共同难度，能检出单向重采样检不出的差异。

---

## 🛠️ 维护工具

```bash
pytest -q -m "not slow"          # 单元测试（无网络、无 GPU 依赖）
ruff check src tests tools        # 静态检查
catface doctor                    # 环境与数据就绪度
python -m tools.audit_species     # 审计语料是否含人脸（新数据源必过）
python -m tools.audit_content     # 审计语料内容类别
python -m tools.parallel_download # 多流并行下载（代理限速时提速约 5 倍）
python -m tools.analyze_corpus    # 语料统计（规模/分辨率/质量/物种构成）
python -m tools.analyze_species_recognition  # 全物种识别统计（跨物种可分性）
python -m tools.run_benchmark_suite  # 一键跑完整基准并出报告
```

### 首次克隆后安装 git 守卫（一次即可）

本仓库用 `git add . && git push` 同步到公开远端，因此以下三类错误一旦推送就**不可撤销**，已做成提交时自动拦截：机器特有绝对路径、凭据赋值、>5 MB 的大文件。

```powershell
.\tools\install-git-hooks.ps1
```

细节与三层防线的分工见 [`docs/HYGIENE.md`](docs/HYGIENE.md)。


## ⏸️ 训练可随时暂停与续训

每个 epoch 结束都会原子化检查点，中断不丢进度、续训轨迹与不中断时一致（状态含优化器动量、调度器位置、采样器 epoch、RNG）：

```bash
# 带时间预算运行，到点自动在 epoch 边界干净停下
python -m tools.train_embedder --backbone dinov2_vits14 --epochs 20 \
  --output artifacts/train/dinov2s-arcface --max-seconds 3600

# 接着上次继续（同一条命令加 --resume）
python -m tools.train_embedder --backbone dinov2_vits14 --epochs 20 \
  --output artifacts/train/dinov2s-arcface --resume
```

| 暂停方式 | 触发 |
|---|---|
| 时间预算 | `--max-seconds <秒>` |
| 信号 | `Ctrl-C` / SIGTERM（在 epoch 边界停，不再中途硬杀） |
| 外部请求 | 在输出目录放 `PAUSE` 文件 |
| 续训 | `--resume` |

也可用 `tools/train_resumable.ps1` 自动循环续训。暂停**不会**被标记成"早停收敛"（用独立的 `pause_reason` 区分），且暂停时保留**最新**权重而非最佳权重——否则续训轨迹会与不中断时不同。

---

## 📄 许可证

MIT，见 [LICENSE](LICENSE)。

使用的数据集各有其许可，详见 `tools/` 与 `catface/data/sources.py` 中的元数据。**Oxford-IIIT Pet 与 CALFW 仅供研究使用**，图像版权归原始来源所有。
