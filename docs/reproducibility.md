# 评测口径与 NVIDIA 移植状态

## 保留的实验条件

- Qwen2.5-VL-7B 和 InternVL3-8B 取 32 帧；MiniCPM-V-4 取 16 帧。
- BF16、greedy decoding（temperature=0）、seed=0。
- LVB 最多生成 32 tokens；VideoMME/MLVU 最多 16 tokens。
- 不输入字幕；VideoMME 的历史提示词仍含 “video and the subtitles”，这是原模板文字，并不表示实际加载字幕。为保持对比一致没有只给 APS 修改模板。
- Qwen 按原生视频尺寸使用历史 28 倍数对齐与 bicubic resize，再关闭 processor 的空间 resize 和二次视频采样；InternVL 以视频通道传入原始帧；MiniCPM 以 K 张 PIL 图像传入，图像占位符之间保留空格。
- Qwen Uniform 使用 float32 linspace 后 round；InternVL/MiniCPM 使用等分区间中心采样。候选不足 K 时使用本模型的同一 Uniform 规则替换整个输入。
- 历史 Qwen/InternVL 使用同一组 BLIP 缓存与 K=32 APS 选帧；不是每个答题模型单独评分。

## 指标

LVB/VideoMME：正确题数 ÷ 全部题数。MLVU：分别计算七个 MCQ task 的准确率，再做等权算术平均。输出同时保留 micro 供审计，但 `main_table_accuracy_percent` 才是主表列。MLVU 的少量冒烟测试没有覆盖七任务，不报告正式 macro。

生成结束不等于成功解析：解析失败数量单列，MLVU/VideoMME 的空选项按错误计算；LVB 保留原始解析器的固定逐题 seed 随机兜底。既有结果中 MiniCPM–MLVU 的这种情况较多。改变解析器需要对所有方法统一重新评估，不能仅改善某一方法。

## NVIDIA 入口做了哪些移植

移除了平台设备变量、国产 GPU 插件、平台 Triton 路径、专用视觉编码分块注入和 transformers 兼容垫片。使用上游 vLLM 的 `LLM`/`SamplingParams` 与 `CUDA_VISIBLE_DEVICES`。推理目录由 CLI 指定，没有绑定机器名、容器名或 NFS 目录。

所选依赖版本参照 [vLLM 0.20.2 包元数据](https://pypi.org/project/vllm/0.20.2/) 与 [上游多模态支持列表](https://docs.vllm.ai/en/latest/models/supported_models/)。普通 NVIDIA wheel 的 PyTorch 依赖不等于旧平台 wheel 的依赖，因此不复制旧版本字符串作为 CUDA 安装命令。

GPU 权重加载、processor、vLLM 视觉 token 预算、长视频显存、特定 NVIDIA 型号上的 attention 后端仍需真实冒烟验证。当前机器上的国产卡不提供 NVIDIA 验证条件。仓库的 CPU CI、静态检查、选帧一致性结果不能当作 CUDA 推理已验证。

## 内存与并行

Qwen–MLVU 历史上下文上限为 127,800，KV cache 配置为每 GPU 10 GiB；其他主表组合通常为 32,768 / 4 GiB，实际以 `configs/main_table.json` 为准。这些值是实验配置，不是“总显存只需要 4/10 GiB”；还需模型、视觉编码和临时张量的显存。

Qwen32B 规模消融配置 TP=2。其他任务默认 TP=1；可直接运行 `aps.infer --tensor-parallel-size 2 --gpus '0,1'` 显式做新的并行配置，并保存独立输出。改变并行度或计算后端可能改变浮点数值，不能承诺逐 token 相同。

## 恢复与完整性

输出 manifest 绑定 annotation、候选、选帧、解析函数、代码、模型配置、权重元信息、版本及运行参数。相同命令可恢复已完成题目；中断留下的未换行 JSONL 尾部在获得目录锁后修复，不修改完整记录。分片只读自己的文件，防止并发读到别人的半行。

汇总必须具备所有预期 `(question_index, method)` 记录，检查题目 ID、gold 和 correct 一致；缺题不产生正式结果。改变模型/方法/帧数/分片参数时不混入原目录。
