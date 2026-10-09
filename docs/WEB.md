# Web 端：猫脸个体识别检索服务

`python -m catface.web` 启动一个搜索页 + JSON API。默认只监听 `127.0.0.1`——**默认不开公网**是
刻意的：这个服务没有用户体系，把绑定地址改成 `0.0.0.0` 之前请先读第 4 节。

## 1. 启动

```bash
pip install -e ".[web]"          # FastAPI + uvicorn，可选依赖组
python -m catface.web --checkpoint artifacts/train/dinov2s-arcface/best.pt \
                      --manifest data/manifests/cat_individuals_manifest.jsonl \
                      --device cuda --port 8000
```

打开 <http://127.0.0.1:8000>，或看交互式 API 文档 <http://127.0.0.1:8000/docs>。

| 参数 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `--checkpoint` | `CATFACE_CHECKPOINT` | `artifacts/train/dinov2s-arcface/best.pt` | 必需，缺失会**启动前**报错而非起一个只能返回 503 的服务 |
| `--manifest` | `CATFACE_MANIFEST` | `data/manifests/cat_individuals_manifest.jsonl` | 定义可检索的身份集合 |
| `--device` | `CATFACE_DEVICE` | `cpu` | `cuda` 可显著降低首屏等待 |
| `--image-size` | `CATFACE_IMAGE_SIZE` | `224` | **必须与 checkpoint 训练时一致** |
| `--host` / `--port` | `CATFACE_HOST` / `CATFACE_PORT` | `127.0.0.1:8000` | |
| `--gallery-root` | `CATFACE_GALLERY_ROOT` | 仓库根 | manifest 中相对路径的解析基准 |
| — | `CATFACE_API_KEY` | 未设置 | 设置后 `/api/search` 需要 `X-API-Key` 或 `Authorization: Bearer` |

## 2. 实测性能（本机 RTX 3060 Laptop，DINOv2-S + ArcFace）

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 启动加载 + 嵌入图库 | **164 s** | 12 644 张图，**每个进程一次**，不是每次请求 |
| 单次查询嵌入 | **32 ms** | 224px + TTA(identity, hflip) |
| 相似度检索 | **0.9 ms** | 12 644 × 512 的精确矩阵乘 |

模型在进程生命周期内只加载一次，所以首屏的 164 秒是唯一一次；此后每次检索约 33 ms。

## 3. 为什么用精确扫描而不是近似索引

图库只有 12 644 条 512 维描述子（约 25 MB），一次精确矩阵乘只要 0.9 ms。换成 ANN 索引会引入
一个 recall 参数，使**报告的数字依赖于索引调参而不是模型本身**——对一个以基准数字为核心的项目，
这是反向的取舍。所以 `/api/search` 明确是穷举扫描。

## 4. 部署注意事项

- **默认绑定回环**。要对外提供服务，请先设置 `CATFACE_API_KEY`，并在前面放一层带 TLS 的反向代理。
- **上传上限 12 MB**，且**边写边判**，超限立即返回 400——不会先把整个请求体读进内存。
- **文件类型靠解码判定，不靠后缀**。后缀只用于提前拒绝；真正的校验是 PIL 能否解码，
  以及描述子是否退化（零范数会被拒绝，因为那样所有余弦都失去意义）。
- **`/healthz` 不依赖模型**，`/api/status` 才表示"能否检索"。这样编排系统能区分
  「进程死了」与「模型加载不了」——后者会一直返回 503，但存活探针仍绿。
- 错误分两类：客户端问题 400/401/404，本项目内部错误 500 且**返回真实原因**（`kind` 字段给出异常类型），
  不吞成通用 500。

## 5. API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/healthz` | 存活探针，不触碰模型 |
| `GET` | `/api/status` | 就绪状态与模型事实（图库规模、维度、骨干、TTA、加载耗时） |
| `GET` | `/api/identities` | 各身份在图库中的图片数 |
| `GET` | `/api/gallery/{image_id}` | 返回图库原图（用于结果缩略图） |
| `POST` | `/api/search` | 上传一张照片，返回排序结果 |

`POST /api/search` 的响应是 Pydantic 模型生成的固定契约（见 `/openapi.json`），字段包含：

- `predicted_identity`、`top_similarity`
- **`margin`** = top-1 相似度 − 最强**不同身份**的相似度。这是本项目刻意暴露的可信度信号：
  错误分析显示 16 个错例的 margin 中位数只有 −0.0569，**全部是贴边误判**。前端把它画成以 0 为中心的
  条形图，负值染红。
- `matches[]`，每项含 `rank`、`identity`、`similarity`、`identity_consensus`
- `timing.embedding_ms` / `timing.search_ms`

## 6. 前端为什么把"做不到什么"放在页面上

一个只展示自信排序结果的检索演示会误导使用者。首页固定渲染三条**实测**边界
（来源见 [`docs/RESEARCH-RETRIEVAL-LIMITS.md`](RESEARCH-RETRIEVAL-LIMITS.md) 与
[`docs/diagnostics/LABEL-COLLISIONS.md`](diagnostics/LABEL-COLLISIONS.md)）：

1. 相似度不等于正确率，hit@1 = 0.9682，且错例全部贴近决策边界；
2. 语料有 137 对同图双标签，同一只猫可能以两个身份名出现且都正确；
3. 身份训练会削弱通用视觉能力（品种 1-NN 从 0.9506 降到 0.7881），不要当通用特征提取器用。

前端是服务端渲染 + 原生 JS，**无构建步骤**：一个没有工具链的 checkout 也能直接跑，
且 JS 失败时降级为可读静态页而不是白屏。

## 7. 容器化部署

```bash
docker compose up --build          # 打开 http://127.0.0.1:8000
```

**模型与图库是挂载进去的，不打进镜像**：86 MB 权重和 25 MB 清单与代码的变更节奏无关，
把模型烧进镜像意味着"回滚模型必须连服务一起回滚"。替换 checkpoint 只需换挂载：

```bash
CATFACE_CHECKPOINT=artifacts/train/dinov2b-arcface/best.pt docker compose up -d
```

镜像设计上的几个有意选择：

| 选择 | 原因 |
|---|---|
| 多阶段构建（build → runtime） | 最终镜像不含编译工具链，既是体积也是攻击面 |
| CPU-only torch（`--index-url …/cpu`） | 默认 wheel 会拉进数 GB 的 CUDA 库而这个镜像用不到；GPU 部署应从 `nvidia/cuda` 基础镜像起 |
| 非 root 用户 `catface` | 能写自己代码的容器，在被利用后就是可改的容器 |
| `HEALTHCHECK` 打 `/healthz` | 该端点**不触碰模型**；用 `/api/status` 会在约 164 s 的模型加载期间把健康容器反复重启 |
| `.dockerignore` 排除 `data/`、`artifacts/`、`.git`、`.kaggle` | 构建上下文从 **27 GB 降到 5.4 MB**；且 `.git` 含完整历史、`.kaggle` 含凭据，而镜像层是永久的 |
| 默认只发布 `127.0.0.1:8000` | 服务没有用户体系，默认开到公网不是安全默认值 |

### 未构建镜像——这一点必须说清楚

**本机没有 Docker（`docker` 不在 PATH 上），因此镜像从未被构建过**，本项目也不声称构建通过。
在没有守护进程的前提下能验证的，是"会让构建失败或悄悄发布错东西"的那一类问题，
由 `tools/check_container.py` 覆盖（并作为 `tests/test_container.py` 运行）：

- 每个本地 `COPY` 源都存在，且没有被 `.dockerignore` 排除（最常见的构建失败原因）；
- 每个 `COPY --from=<stage>` 都指向已声明的 stage；
- compose 的 `build.target` 在 Dockerfile 中存在；
- entry point 模块可导入（容器内执行的就是这条命令）；
- 权重与语料没有被打进镜像；
- 健康检查用的是存活端点而非就绪端点。

这是**静态检查，不是构建的替代**。`.dockerignore` 的匹配语义是**保守近似**——真实 Docker 实现
会覆盖它；写这套近似时用正反例测出并修掉了两个自己的 bug（尾斜杠模式漏掉目录本身、
把路径里的 `!` 当成否定符）。

## 8. 测试

```bash
pytest tests/test_web.py -q
```

测试用**假 embedder**（固定向量），因此无需权重、无需 GPU，也不依赖网络。覆盖的是协议问题：
拒绝什么、报告什么、近并列是否可见。其中一条测试断言的是**不变量**而非某个结果——见
`test_identity_aggregation_can_only_lower_an_identity_score`：均值不可能超过被平均的最大值，
所以"身份级聚合"在结构上无法提升 top-1，这条性质比"哪个身份赢"更值得钉住。
