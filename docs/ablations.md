# 消融复现

所有任务只生成命令，除非显式加 `--execute`。所有默认值来自主表；只改变被消融的轴。`configs/ablations.json` 同时包含用于对照的主表点，共 82 个配置条目，不意味着需要重跑 82 个不同实验。

调度器以完整实验参数生成输出目录；完全相同的配置可跨消融轴复用，新任务会检查 manifest 后恢复或直接汇总已完成结果。已有旧结果不能仅按文件夹名当作完成；导入旧结果应额外校验其参数/输入。历史结果在 `reports/ablations/` 单独保留。

## α/β

Qwen7B、32 帧、BLIP、λ=0.5；仅 LVB/VideoMME。3×4×2=24 个组合。

```bash
python -m aps.experiments --config configs/ablations.json --axis alpha_beta --gpus '0;1;2;3'
```

## 帧数

Qwen7B，8/16/32/64 帧，APS 与 Uniform，LVB/VideoMME。64 帧按历史 manifest 使用 49,152 上下文和 4 GiB KV cache，仍保留原始图像预处理。32 帧是主表对照点。

```bash
python -m aps.experiments --config configs/ablations.json --axis frames --gpus '0;1;2;3'
```

## 模型规模

```bash
python scripts/download_models.py --models qwen3b qwen32b
python -m aps.experiments --config configs/ablations.json --job <EXACT_JOB_ID> --gpus '0,1;2,3'
```

32B 的配置 TP=2、每卡 6 GiB KV cache，因此选择其具体 job ID 时使用双卡组。3B/7B 的配置 TP=1，使用 `--gpus '0;1;2;3'`。不要把含 TP=1 和 TP=2 的整组任务用同一 GPU 分组一口气启动；调度器会报错。

## 编码器

```bash
python scripts/download_models.py --provider huggingface --models clip blip2
CUDA_VISIBLE_DEVICES=0 python -m aps.score \
  --dataset longvideobench --annotation data/longvideobench/questions.json \
  --video-root data/longvideobench/videos --model models/clip-vit-base-patch32 \
  --scorer clip --candidates cache/longvideobench/blip/frames.json \
  --output-dir cache/longvideobench/clip --num-shards 1
```

同样参数再加 `--merge` 得到 `scores.json/frames.json`。BLIP-2 改为 `--scorer blip2 --model models/blip2-itm-vit-g --output-dir cache/longvideobench/blip2`。VideoMME 替换 dataset 和目录。所有编码器必须复用并核验 BLIP 候选网格。

```bash
python -m aps.experiments --config configs/ablations.json --axis encoder --gpus '0;1;2;3'
```

## 区间划分

- `full`：混合常数/线性动态规划。
- `equal_length`：先求完整 COMPS 的区间数 J，再按候选位置等长划分为 J 个连续区间；预算分配及区间内 Top-K 不变。
- `linear_only`：DP 只保留线性模型及原长度上限。
- `constant_only`：DP 只保留常数模型，仍然没有区间最大长度限制。

Qwen7B 和 MiniCPM，各自使用主表 32/16 帧，在三个数据集上运行；每模型三个新变体共九组，full 是主表对照。

```bash
python -m aps.experiments --config configs/ablations.json --axis partition --gpus '0;1;2;3'
```

## 复杂度惩罚 λ

InternVL：0、0.25、0.5、0.75，三个数据集；Qwen：λ=0，只跑 LVB/VideoMME、32 帧。α、β 继续取对应主表值。

```bash
python -m aps.experiments --config configs/ablations.json --axis penalty --gpus '0;1;2;3'
python -m aps.experiments --config configs/ablations.json --axis penalty_zero --gpus '0;1;2;3'
```

λ=0 只去掉复杂度项，不移除平滑、噪声归一化、最短段长、线性最长段长、DP 拟合代价、预算分配或 Top-K。它不是 Uniform，也不保证固定区间数。
