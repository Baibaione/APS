# 对比方法

本仓库自带 APS 与原模型 Uniform 推理。其他方法的历史汇总保存在 `reports/main_table.json`，不把第三方整仓、环境垫片、下载的模型和内部调度脚本复制到 APS 包内。

| 方法 | 来源 / 在历史实验中的作用 |
|---|---|
| AKS | https://github.com/ncTimTang/AKS ，共享 BLIP 分数 |
| WFS | 历史实验中的 WFS 流程；这里保留结果，未把其独立实现声明为 APS 代码 |
| DIG | https://github.com/Jialuo-Li/DIG ，历史实验采用共享 Qwen 奖励 |
| FOCUS | https://github.com/NUS-HPC-AI-Lab/FOCUS ，历史适配采用 BLIP-ITM Large |
| QCA | https://github.com/hktk07/QCA ，包含其额外视觉特征处理 |
| RIDGE | https://github.com/Xssq999/RIDGE ，Qwen3-8B 生成共享辅助权重，答题仍为主表三个模型 |

这些方法的完整复现依赖各自官方许可、参数和独立输入，不应仅将 APS 选帧替换成一个同名函数就宣称官方复现。当前交付的可运行主线为 APS/Uniform 及 APS 消融；各基线旧结果作为参考证据。
