# APS

APS 面向长视频问答，在固定帧预算下根据问题相关性曲线选择视频帧。流程为：**相关性打分 → 自适应区间划分（COMPS）→ 区间预算分配 → 区间内 Top-K → 视频问答评测**。

历史实验代码、结果中的 `dps` 是同一个方法的旧名称；本仓库对外统一使用 **APS**。这里不包含训练流程，三个视觉语言模型均直接进行推理。

> 本仓库整理了实际实验算法、正式选帧和配置。历史结果来自 MetaX GPU / FlagOS。下文推理入口面向 **NVIDIA CUDA + 上游 vLLM**，已进行代码检查和 CPU 协议测试，尚未在 NVIDIA GPU 上完成端到端验证，不能把已有结果称为 NVIDIA 实测结果。

## 目录

```text
src/aps/                 APS 算法、打分、NVIDIA 推理、指标与实验调度
configs/main_table.json  三个模型 × 三个数据集的主表配置
configs/ablations.json   参数、编码器、规模、帧数、区间划分和惩罚系数消融
assets/selections/       6 份正式 APS 选帧：3 个数据集 × 16/32 帧
assets/question_ids/     正式评测问题顺序，不包含问题正文和答案
assets/question_manifest/  视频名与问题哈希，用于核对问题身份
assets/input_manifest.json  历史标注、分数和候选帧的 SHA-256
reports/                脱敏后的结果、源文件哈希和验证记录
scripts/                模型下载、数据导入、MLVU 转换、发布目录检查
examples/               无模型、无数据依赖的 CPU 示例
third_party/            第三方代码来源与许可证
```

模型权重、视频、完整数据集、打分缓存、原始预测和运行日志不进入 Git。打分缓存可从已有实验导入，也可按下述步骤重新生成。

## 1. 先运行 CPU 示例

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python examples/quickstart.py
pytest -q
```

最小使用方式：

```python
from aps import APSConfig, select_frames

selected = select_frames(
    scores,                # 一题的候选帧相关性分数，按时间排列
    original_frame_ids,    # 对应的原始视频帧编号，不是秒数
    APSConfig(budget=32, alpha=1.0, beta=0.6, penalty=0.5),
)
```

## 2. NVIDIA 推理环境

建议使用 Linux x86_64、Python 3.11，以及支持 BF16 的 NVIDIA GPU。先确认 `nvidia-smi` 正常。使用全新环境，让 vLLM 安装匹配的 CUDA PyTorch；不要从原国产卡容器复制 torch、Triton 或平台插件。

```bash
conda create -n aps-nvidia python=3.11 -y
conda activate aps-nvidia
python -m pip install --upgrade pip
python -m pip install -r requirements-nvidia-lock.txt
python -m pip install -e '.[test,download]'
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

固定候选环境：上游 `vllm==0.20.2`、`transformers==4.57.1`、`decord==0.6.0`。`requirements-nvidia.txt` 是直接依赖，`requirements-nvidia-lock.txt` 已按 Linux x86_64 / Python 3.11 解析并锁定 189 个依赖。这是供 NVIDIA 验证的依赖组合，不是原 FlagOS 环境的版本锁文件，也不代表 GPU 运行验证。CUDA 驱动须满足所安装 PyTorch/vLLM wheel 的要求，见 [vLLM GPU 安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。

Qwen 的历史协议保留视频原生分辨率，MLVU 的最大上下文设为 127,800，显存需求明显高于其他组合。遇到显存不足时使用更大显存 GPU 或显式张量并行；**不要通过静默降分辨率或少取帧来获得表面可运行的主表结果**。详见 [推理与复现口径](docs/reproducibility.md)。

## 3. 下载三个答题模型

默认从 ModelScope 下载：

```bash
python scripts/download_models.py --models qwen7b internvl minicpm
```

| 模型别名 | 答题模型 | 主表帧数 |
|---|---|---:|
| qwen25vl | Qwen2.5-VL-7B-Instruct | 32 |
| internvl | InternVL3-8B | 32 |
| minicpm | MiniCPM-V-4 | 16 |

已有模型可直接放入或链接到 `models/` 下相应目录。要固定具体模型修订，可使用下载脚本的 `--revision`；实际推理会记录模型配置哈希、权重文件元信息和依赖版本。

APS **不需要 Qwen3-8B 辅助权重模型**；那属于 RIDGE 基线。本方法主表使用共享的 BLIP-ITM Large 相关性缓存。

## 4. 数据与分数缓存

正式协议是 LongVideoBench 1,337 题、VideoMME 2,700 题、MLVU 七个选择题任务共 2,174 题，均不输入字幕。

如果已有历史实验目录，直接导入，可保持标注、问题顺序和浮点分数完全一致：

```bash
python scripts/import_legacy.py --source /path/to/previous/AKS
```

目录如下：

```text
data/<dataset>/questions.json
data/<dataset>/videos/...
cache/<dataset>/blip/scores.json
cache/<dataset>/blip/frames.json
```

`<dataset>` 为 `longvideobench`、`videomme`、`mlvu`。视频目录只创建链接，不复制完整视频。输入导入会校验历史 SHA-256；哈希不匹配时停止。

从零准备数据、MLVU 转换以及重新打分见 **[数据准备文档](docs/data.md)**。复用选帧必须同时匹配标注、问题顺序、候选帧和视频解码网格；不能只根据题目数量判断缓存可用。

## 5. APS 选帧

以 LVB、32 帧为例：

```bash
python -m aps.select \
  --scores cache/longvideobench/blip/scores.json \
  --frames cache/longvideobench/blip/frames.json \
  --annotation data/longvideobench/questions.json \
  --output cache/selected/lvb_aps_k32.json \
  --budget 32 --alpha 1.0 --beta 0.6 --penalty 0.5 \
  --workers 8 --details
```

`--details` 额外保存分区边界、每区预算和候选位置，便于案例可视化。主表参数如下：

| 数据集 | α | β | λ | 平滑半径 | 最小区间长度 | 线性区间最大长度 |
|---|---:|---:|---:|---:|---:|---:|
| LVB | 1.0 | 0.6 | 0.5 | 2 | 3 | 80 |
| VideoMME | 1.0 | 0.4 | 0.5 | 2 | 3 | 80 |
| MLVU | 1.0 | 0.4 | 0.5 | 2 | 3 | 80 |

Qwen 和 InternVL 使用相同分数及参数，复用 32 帧选帧；MiniCPM 使用 16 帧选帧。因此主表为 **6 份选帧、9 组 APS 答题评测**。

导入历史缓存后，可用 `python scripts/verify_selections.py --workers 8` 重做全部 12,422 次选帧比对。本次整理的验证记录见 `reports/selector_parity.json`。

## 6. 运行单组推理

先预检查，再跑少量题，最后全量运行。以下使用仓库中的正式 LVB 选帧。

```bash
python -m aps.infer check \
  --dataset longvideobench --model models/Qwen2.5-VL-7B-Instruct \
  --candidates cache/longvideobench/blip/frames.json \
  --selected assets/selections/longvideobench/aps_k32.json \
  --methods aps --nframes 32 --gpus '0' --limit 8 \
  --run-dir outputs/qwen_lvb_smoke

python -m aps.infer run \
  --dataset longvideobench --model models/Qwen2.5-VL-7B-Instruct \
  --candidates cache/longvideobench/blip/frames.json \
  --selected assets/selections/longvideobench/aps_k32.json \
  --methods aps --nframes 32 --gpus '0' --limit 8 \
  --run-dir outputs/qwen_lvb_smoke
```

预检查解码所选范围的视频并检查帧网格，不加载大模型权重。推理前会加载 processor；`check` 不能替代 GPU 冒烟测试。全量运行移除 `--limit 8`，并换成独立的 `--run-dir outputs/qwen_lvb_full`。

多 GPU 数据并行：`--gpus '0;1;2;3'`，每张卡独立加载模型、处理不同题目。双卡张量并行：`--gpus '0,1' --tensor-parallel-size 2`；两组双卡：`--gpus '0,1;2,3' --tensor-parallel-size 2`。引号不可省略。

结果位于 `--run-dir`：`manifest.json`、`rank*.jsonl`、`rank*.log`、`summary.json`。重新运行**相同命令**即可恢复；参数、分片数或输入变化时须使用新目录，防止混入旧结果。

## 7. 主表九组实验

先输出启动计划（不使用 GPU）：

```bash
python -m aps.experiments --config configs/main_table.json --gpus '0;1;2;3' --methods aps
```

九组各跑 8 题检查：

```bash
python -m aps.experiments --config configs/main_table.json \
  --gpus '0;1;2;3' --methods aps --limit 8 --execute
```

检查日志后全量运行：

```bash
python -m aps.experiments --config configs/main_table.json \
  --gpus '0;1;2;3' --methods aps --execute
```

单组可加 `--job qwen25vl_longvideobench`，其他 ID 见 JSON。要同时运行 Uniform，改为 `--methods aps,uniform`，此时是九组模型/数据集组合、每组两种选帧方法，共十八项指标。调度脚本顺序执行所选组合，每个组合内部按 GPU 组并行；没有绑定固定机器或 IP。

## 8. 消融实验

统一配置见 [configs/ablations.json](configs/ablations.json)，具体步骤见 [docs/ablations.md](docs/ablations.md)。包含：

- α/β：α∈{0.5,1.0,1.5}，β∈{0.4,0.5,0.6,0.7}；Qwen7B、LVB/VideoMME。
- 帧数：8/16/32/64；Qwen7B，APS/Uniform，LVB/VideoMME。
- 模型规模：Qwen2.5-VL 3B/7B/32B；APS/Uniform，LVB/VideoMME。
- 编码器：BLIP、CLIP、BLIP-2；Qwen7B、LVB/VideoMME。
- 区间划分：full、equal_length、linear_only、constant_only；Qwen 和 MiniCPM、三个数据集。
- λ：InternVL 的 0/0.25/0.5/0.75、三个数据集；Qwen 的 λ=0、LVB/VideoMME。

## 9. 指标和已有结果

LVB、VideoMME 使用逐题准确率；**MLVU 主指标使用七任务准确率的宏平均**。不得把 MLVU 的 micro 数值放入同一张主表。

历史 APS 结果如下，完整 8 种方法 × 9 组结果保存在 [reports/main_table.json](reports/main_table.json)。这些是已有国产卡实验的汇总，不是本次 NVIDIA 移植的测量结果。

| 模型 | 帧数 | LVB | VideoMME | MLVU Macro |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 32 | 63.80% | 62.33% | 68.05% |
| InternVL3-8B | 32 | 63.58% | 66.85% | 74.17% |
| MiniCPM-V-4 | 16 | 58.12% | 57.63% | 58.21% |

解析器与原实验保持一致：LVB 无法解析时使用固定逐题种子的随机兜底；VideoMME/MLVU 无法解析出选项时计错，并单列数量。MiniCPM 有部分回答直接输出答案文本，不能在某个方法上单独补解析而不重评所有对比方法。

## 10. 仓库发布

```bash
pytest -q
python scripts/check_release.py
python -m build
```

本目录可直接作为独立 Git 仓库。初始化及推送步骤见 [docs/release.md](docs/release.md)。项目论文题目、作者、引用信息和自有代码许可证应使用最终确认内容；没有虚构占位 DOI 或替你选择开源授权。

第三方来源与现有版权声明见 [third_party/NOTICE.md](third_party/NOTICE.md)。其他选帧基线的方法仓库和对比口径见 [docs/baselines.md](docs/baselines.md)。
