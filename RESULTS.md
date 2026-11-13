# Narl CPU 实验报告

在 Apple M2（8 GiB）上，以 19,424 参数的 FP32 Llama 和合成 token-choice 任务，测量权重切换时的 KV 重算、同样本重分组和真实双进程吞吐。上游为 nanoRL `793e9b5`，保留 MIT 许可。实验 A 和正式在线运行的源码为 `de59470`，实验 B 为 `139b90c`。运行配置、快照哈希、随机种子、模型和软件版本随原始结果保存。环境为 Python 3.11.15、PyTorch 2.8.0、Transformers 4.57.6。[复跑入口](README.md) · [完整依赖版本](runs/requirements-cpu-resolved.txt) · [主机记录](runs/cpu-host.txt)

## A. 权重切换后重算 KV

每次固定同一个历史，G=8，已经缓存的分支后缀为 16 token。第 17 个 token 由旧策略采样，权重切换后作为首个 decode 输入。旧快照为初始权重，新快照来自 12 步同步 GRPO。每个配置预热 3 次并测量 15 次。恢复时间从开始恢复到首个 logits，权重加载单列。

| Prompt tokens | 保留旧 KV，ms | 逐条重算，ms | 共享 prompt 重算，ms | 保留旧 KV 平均 KL，nats |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.426 [0.351, 0.446] | 1.925 [1.644, 2.108] | 2.668 [2.313, 3.247] | 0.002014 |
| 128 | 0.568 [0.474, 0.602] | 7.118 [6.669, 8.178] | 3.853 [3.368, 4.354] | 0.002017 |
| 384 | 0.926 [0.746, 1.024] | 49.224 [45.118, 51.034] | 10.020 [8.973, 12.115] | 0.002436 |

括号为最小值和最大值。KL 为逐条重算分布到候选分布的方向，沿相同的 teacher-forced 历史测量。共享重算在 384-token 配置将恢复耗时降低 79.6%，32-token 配置则更慢。共享路径只重算一次 prompt，分支后缀各自重算，物理复制 prompt KV 的耗时已包含在恢复时间中。

同权重缓存与完整前向对齐。共享与逐条重算的最大 logits 误差为 `1.19e-7`，192 个位置的贪心预测一致。用相同的独立随机数实际采样，两条路径的 192 个 token 一致。测试还覆盖左 padding、位置偏移、EOS 后的统计屏蔽和分支隔离。[汇总](runs/kv-replay-release/summary.json) · [原始耗时](runs/kv-replay-release/timings.csv) · [固定输入](runs/kv-replay-release/inputs.json) · [图表](runs/kv-replay-release/replay.png)

## B. 同一批 completion 按版本重分组

使用一个真实 GRPO 训练轨迹的五个检查点，每个 prompt 从两份权重各采 8 条。比较两个同版本组与两个各含 4 新 4 旧的组。两臂共享全部 completion ID、token、mask、行为 logprob 和 reward，重分组后按 ID 对齐 advantage，随后用相同 learner 和 loss 分母计算梯度。16 个 prompt、3 个采样种子、每种子 8 次重分组，共 2,304 条唯一 completion、120 个配对比较。同权重对照使用独立采样流。

| 快照对 | 共同历史上的平均 KL，nats | 平均绝对 advantage 变化的中位数 | 梯度余弦中位数 | 零信号组比例中位数，同版 → 混版 |
| --- | ---: | ---: | ---: | ---: |
| v4 / v32 | 0.1062 | 0.249 | 0.963 | 34.4% → 28.1% |
| v12 / v32 | 0.1153 | 0.235 | 0.975 | 28.1% → 25.0% |
| v24 / v32 | 0.0852 | 0.231 | 0.984 | 21.9% → 21.9% |
| v31 / v32 | 0.0030 | 0.219 | 0.981 | 18.8% → 25.0% |
| v32 / v32 | 0 | 0.232 | 0.975 | 21.9% → 25.0% |

三对较高漂移快照中，每个种子内相对配对同权重对照的 advantage 变化中位数均为正，范围为 0.004–0.054。低漂移对接近同权重对照。各条件的变化范围有重叠，梯度余弦也随重分组而变。这个结果来自一条训练轨迹，支持区分重分组本身的波动和策略漂移，尚无单调的滞后效应。两次运行的 completion、分组、训练 trace 和 prompt 文件逐字节一致。[汇总](runs/regroup-release/summary.json) · [样本](runs/regroup-release/completions.jsonl) · [配对明细](runs/regroup-release/pairs.jsonl) · [图表](runs/regroup-release/regroup.png)

## C. 真实双进程在线对照

实验 A 两条重算路径通过数值对齐，因此选择逐条重算与共享 prompt 重算做在线主对照。actor 和 learner 各一个 CPU 进程，各一条 PyTorch 计算线程。每个 arm 固定队列容量 2、一个已发布快照槽、发布频率每个准入组一次、滞后上限 1。两臂对每个种子使用相同初始权重、训练 prompt 流和 18,432 个生成 token 的预算，保留自由采样。生成长度按 16、24、32 token 循环，奖励来自末 token。每四个 token 检查新权重，每组最多切换一次。正式运行前两个短跑 arm 均观察到跨版。固定输入测试核对了跨版前后逐 token 行为 logprob、token、reward 和恢复计算量。[短跑](runs/kv-online-swap-pilot/summary.json) · [固定换版测试](tests/test_narl_kv_online.py)

| 种子 | 换版恢复中位数，ms | 有 advantage 的训练 tokens/s | 消费组/s | 过期丢组 | 最终留出 reward |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.89 → 3.79 | 1575 → 1646 | 14.21 → 14.08 | 18 → 21 | 0.25 → 0.25 |
| 1 | 7.56 → 3.86 | 1723 → 1622 | 15.72 → 14.91 | 16 → 16 | 0.25 → 0.25 |
| 2 | 7.42 → 3.90 | 1708 → 1650 | 16.80 → 17.18 | 11 → 16 | 0.25 → 0.25 |

箭头为逐条重算到共享重算。墙钟含 actor 启动、评估和写日志。共享重算的每种子恢复时间快 1.90–2.08 倍，端到端训练 token 吞吐的配对差异为 −101 到 +71 token/s，方向不一致。六个运行起点和终点的留出集 reward 均为 0.25，seed 1 的两臂早期曾跌到 0。该任务和预算下未观察到学习收益。

三种子记录了 371 次在途换版。版本号在每个准入组后递增，包括梯度为零的发布。逐组 trace 中，202 次在途切换跨过至少一个非零梯度更新，其中 109 次所在的组实际被消费。所有消费组的 lag 都在 0–1，98 个过期组被拒绝。审计核对了 token 预算、行为版本、准入、丢组、固定评估记录及每次权重切换。生成吞吐、有效组吞吐、零信号比例、排队和更新耗时、消费组与丢弃组的长度及 reward 分布保存在审计结果。压力运行分别观察到最长约 1.86 秒和 0.16 秒的 producer put，两个 arm 各拒绝 4 个过期组。另以相同队列、快照槽和发布频率运行 `τ=2` 的短对照，实际消费了 lag 2 的组。[正式配置](runs/online-release/protocol.json) · [正式审计与分布](runs/online-release/audit.json) · [学习和吞吐图](runs/online-release/online.png) · [压力审计](runs/online-backpressure/audit.json) · [τ=2 审计](runs/online-tau2/audit.json)

## Profiling、部署与验收

- **CPU profiling。** 384-token 回放中，SDPA 占恢复回放 CPU 时间的 85.2%（逐条）和 60.3%（共享）。在线共享路径的 KV 复制中位数约 0.024 ms，actor 拉取权重约 0.20 ms，learner 更新约 15–16 ms。各阶段可重叠。Apple M2 的 PyTorch 检测到 0 个 CUDA 设备，CUDA 微基准未触发。[CPU profile](runs/kv-replay-release/profile.json) · [在线阶段数据](runs/online-release/audit.json)
- **基线和正确性。** 同步与真实双进程异步基线各完成 12 步。固定同一批样本在零滞后下逐位对齐参数更新，排队后变成超龄的组在消费时被拒收。另有五项 KV 测试和一项固定换版概率测试通过。[CPU 基线](runs/cpu-baseline-repro/summary.json) · [测试入口](README.md)
- **Kubernetes。** 用 kind v0.33.0、Kubernetes v1.37.0，在单 Pod 内执行原 KV replay 入口，每种策略每档测 5 次。最终复跑镜像的 11 个源文件与提交 `272f8d6` 的 SHA-256 全部一致。Job 在 27 秒后达到 `Complete=True`，Pod 退出码 0，45 条计时、日志和结果全部可取回。验证后删除临时集群。[Job 状态](runs/k8s-kv-replay.dPMiEh/job.json) · [Pod 状态](runs/k8s-kv-replay.dPMiEh/pods.json) · [日志](runs/k8s-kv-replay.dPMiEh/job.log) · [镜像源码核验](runs/k8s-kv-replay.dPMiEh/source-verification.json) · [结果](runs/k8s-kv-replay.dPMiEh/c958688d-35d5-4788-a867-5209b17aa4c5/summary.json)
