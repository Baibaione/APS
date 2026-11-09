# 数据与缓存

## 历史实验的精确输入

推荐运行 `python scripts/import_legacy.py --source /path/to/previous/AKS`。脚本只导入每个数据集的 `include_frame_idx.json`、BLIP `scores.json`、`frames.json`，并链接本地视频。

整理交付的独立 `APS-reproduction-inputs.tar.gz` 是本地复现输入备份，放在仓库目录之外，不随代码提交。解压后可把其中的 `AKS/` 传给同一个导入脚本；视频需另行下载并链接。该包包含历史标注与相关性缓存，没有模型权重或视频。将来若要发布此包，应保留数据来源与适用许可，不能将第三方数据视为 APS 自有代码。

## 从公开来源重新准备

- [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench)：下载验证集标注及对应视频。
- [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME)：下载测试标注及视频。历史评测使用 AKS 的 `include_frame_idx.json`，不是随意换用另一个标注版本。
- [MLVU 官方数据](https://huggingface.co/datasets/MLVU/MVLU)：使用 `MLVU/json/1_plotQA.json` 到 `7_topic_reasoning.json` 七个 MCQ 任务，以及对应视频文件；不包含开放式生成任务。获取权限和下载方式以数据提供方为准。
- 历史 AKS 标注来源：[ncTimTang/AKS](https://github.com/ncTimTang/AKS)，历史代码修订 `b0b8a58fedf1d05d78151e2969cecde60e83721d`。LVB 历史标注与原始验证集有一题内容差异，因此以本仓库记录的哈希为准。

下载后组织视频为：

```text
data/longvideobench/videos/<video_path>
data/videomme/videos/<videoID>.mp4
data/mlvu/videos/1_plotQA/<video_name>
data/mlvu/videos/2_needle/<video_name>
... 直到 7_topic_reasoning/
```

本仓库不自动下载完整视频，避免把已经下载的数据再复制一份。

MLVU 原始 JSON 转换：

```bash
python scripts/convert_mlvu.py --json-dir /path/to/MLVU/json \
  --output data/mlvu/questions.json
```

转换程序按照仓库中记录的问题文本哈希及视频名匹配到原问题顺序，若数据源发生变化则报错，不根据行号盲目拼接。LVB/VideoMME 接受原评测 JSON schema；从 parquet 导出时必须保留如下字段和问题顺序：

| 数据集 | ID | 视频字段 | 问题/选项/答案 | 分组字段 |
|---|---|---|---|---|
| LVB | `id` | `video_path` | `question`, `candidates`, `correct_choice`（从 0 开始） | `duration_group`, `question_category` |
| VideoMME | `question_id` | `videoID`（不带扩展名） | `question`, `options`（带 A./B. 等标签）, `answer`（字母） | `duration`, `domain`, `sub_category`, `task_type` |
| MLVU | `question_id` | `video_relpath`, `video_name` | `question`（已含选项）, `candidates`, `answer`（字母） | `task_type` |

检查/安装一个新下载的标注：

```bash
python scripts/prepare_annotations.py --dataset longvideobench \
  --input /path/to/include_frame_idx.json --output data/longvideobench/questions.json
```

若确实选择不同标注版本，可使用 `--allow-new-protocol`；随后必须重新打分和选帧，不能复用正式选帧或宣称精确复现历史数字。即使只有 JSON 序列化改变，也会改变文件哈希；精确路径应直接导入原始字节。

## 重新打分

打分模型与答题模型独立。主表三个模型共用一份 BLIP-ITM Large 分数。编码器消融改变打分模型，其余选帧参数不变。需要重算分数时：

```bash
python scripts/download_models.py --provider huggingface --models blip
CUDA_VISIBLE_DEVICES=0 python -m aps.score \
  --dataset longvideobench --annotation data/longvideobench/questions.json \
  --video-root data/longvideobench/videos --model models/blip-itm-large-coco \
  --scorer blip --output-dir cache/longvideobench/blip \
  --shard-id 0 --num-shards 1
python -m aps.score \
  --dataset longvideobench --annotation data/longvideobench/questions.json \
  --video-root data/longvideobench/videos --model models/blip-itm-large-coco \
  --scorer blip --output-dir cache/longvideobench/blip \
  --num-shards 1 --merge
```

答题模型默认 ModelScope 下载；打分模型这里显式使用其 Hugging Face 官方仓库。若已有 ModelScope 镜像或本地模型，直接给 `--model` 本地目录即可。脚本不会偷偷切换模型或下载来源。

多卡打分可分别启动 `--shard-id 0..N-1 --num-shards N`，每个进程设置自己的 `CUDA_VISIBLE_DEVICES`。同一输出目录的分片总数、batch size、模型和标注保持不变；分片结束后执行 `--merge`，缺题或重复题时拒绝合并。

候选网格严格沿用历史实现：令 `stride=int(fps)`，候选为 `j*stride`，其中 `j=0..total_frames//stride-1`。这是近似每秒一帧；29.97 FPS 会用 stride=29，并不是精确时间戳 1 FPS。重新解码的视频如果总帧数或 FPS 不同，不得继续复用缓存。

BLIP 重新打分入口基于 HF `BlipForImageTextRetrieval`、FP32、384×384 PIL bicubic、CLIP mean/std、35-token 文本截断和 ITM 正类概率。MLVU 的 BLIP 分数保留历史 6 位小数规则。LVB/VideoMME 的历史缓存由原先 AKS/LAVIS 路径得到；**HF 新打分与旧缓存没有逐帧数值等价证明**，要精确复现必须导入旧缓存。不能因为模型名称相同就覆盖历史缓存。

CLIP 使用 `openai/clip-vit-base-patch32` 图文特征余弦相似度。BLIP-2 使用 `Salesforce/blip2-itm-vit-g`、ITM 正类概率、FP32、32-token 问题截断。编码器消融仅在 LVB/VideoMME 上运行，问题不追加候选答案；MLVU 本身的 `question` 字段已含候选，代码不会再次追加。
