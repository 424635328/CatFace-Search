# 一次推送就过 CI：推送前自检清单

目标很实际——**别在 CI 上花第二个回合**。一次失败的 CI 代价是：推送 → 排队 → 失败 → 修 → 再推。本项目第一次真实 CI 运行有 3 个 job 失败，两个来回浪费约 45 分钟，而**全部失败在推送前 2 分钟内本地就能发现**。

## 一条命令

```powershell
.\tools\preflight.ps1              # 约 100 秒（含完整测试）
.\tools\preflight.ps1 -SkipTests   # 约 2 秒（只查 lint / 配置 / 跟踪状态）
.\tools\preflight.ps1 -CloneCheck  # 额外做全新克隆验证（最彻底）
```

它就是 CI 的四类检查，**带完全相同的 flag**。

## 为什么必须是"完全相同"，而不是"差不多"

本项目第一次 CI 失败的 3 个 job，**全部源于本地检查与 CI 接近但不相同**：

| 失败 job | 我本地的检查 | CI 的检查 | 差别导致的后果 |
|---|---|---|---|
| lint | `ruff check --select E,F` | `ruff check src tests tools`（我在 `pyproject.toml` 声明的完整规则集 E,F,W,I,N,UP,B,C4,SIM,RET,ARG,PTH,RUF） | 本地干净，CI 报 **103 个错误** |
| test（3.9 与 3.11） | 在工作区里跑 `pytest` | 在**全新克隆**里跑 `pytest` | 本地 265 passed，CI 报 `ModuleNotFoundError: No module named 'catface.data'` |

**教训**：本地"跑了测试/跑了 lint"不等于"跑了 CI 会跑的东西"。**flag、工作目录、文件可见性都是检查的一部分。**

## 三条最容易漏的坑（按隐蔽程度排序）

### 1. 本地存在 ≠ 已提交（最隐蔽，唯一只能靠克隆发现）

一次 `.gitignore` 里无前导斜杠的 `data/`，把 `src/catface/data/` 整个包（6 个模块）挡在了版本控制外。本地测试、本地 lint、本地 import **全部正常**——文件就在工作区。只有别人克隆才崩。

```powershell
# 推送前自查：有没有源码没被跟踪
git ls-files --others --exclude-standard | Select-String "^(src|tests|tools)/"
```

`tools/preflight.ps1` 里每一步都会查这个，因为它的失败是本地的盲区。

### 2. lint 抄了别人的 flag（第二隐蔽）

`ruff` 默认**尊重 `.gitignore`**。被忽略的文件不会被 lint。所以"被 gitignore 误伤的源码包"同时逃过了版本控制和 lint——它一进版本控制，ruff 立刻报出 12 个此前从未被检查过的错误。

```powershell
# 覆盖被忽略的路径，消除盲区（CI 也用这个 flag）
ruff check --no-respect-gitignore src tests tools
```

### 3. 声明支持多个 Python 版本却只用最新的开发

`pyproject.toml` 声明 `requires-python = ">=3.9"`，而我在 3.11 上开发。`@dataclass(slots=True)` 是 3.10+ 才有的参数，导入即失败。CI 的 3.9 job 正确地抓到了。

```powershell
# 有 Python 3.9 时确认一次
uv python install 3.9
python3.9 -c "import sys; sys.path.insert(0,'src'); import catface.pipeline"
```

`tests/test_repo_hygiene.py` 现在有一条守卫：扫描 `requires-python` 的下限，发现代码里用了更新的构造就失败。

## 推送前 5 步（不用脚本时）

```powershell
# 1. lint，用 CI 的 flag
ruff check --no-respect-gitignore src tests tools

# 2. 测 试，用 CI 的标记与覆盖率参数
pytest -q -m "not slow" --cov=catface

# 3. 有没有源码没被跟踪（本地唯一盲区）
git ls-files --others --exclude-standard | Select-String "^(src|tests|tools)/"

# 4. 声明的最低 Python 版本能导入
python3.9 -c "import sys; sys.path.insert(0,'src'); import catface.pipeline"

# 5. 配置仍可解析
python -c "from catface.config import PipelineConfig; PipelineConfig.from_yaml('configs/default.yaml')"
```

或者直接 `.\tools\preflight.ps1`。

## 顺带：这些守卫都是"被咬过"才有的

| 守卫 | 由哪次失败催生 |
|---|---|
| `test_every_python_source_file_is_tracked` | `.gitignore` 藏了 `src/catface/data/` |
| `test_no_source_directory_is_ignored` | 同上（区分"没加"与"被忽略"） |
| `test_project_owned_ignores_are_anchored` | 同上（钉住无前缀模式这个具体错误） |
| `test_no_construct_newer_than_the_declared_floor` | `dataclass(slots=True)` 在 3.9 上失败 |
| `TestNoMachineSpecificPaths` 等 | 硬编码路径 / 凭据 / 大文件 |
| `tools/git-hooks/pre-commit` | 公开仓库 + `git add . && git push` 的组合 |

**每一条都对应一次真实的失败**，没有一条是"预防性地"写出来的——这也是它们有效的原因。
