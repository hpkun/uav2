# Training metrics logging-only 验收

日期：2026-10-13。代码基线：`341e4ae51f8dce857e5f3d30e66f959a17e95a14`。

## 原因与行数

原 MAPPO 和 MADSAC runner 都在每个 vector step 写 training_metrics.jsonl。
24 个环境的 3M transitions 对应 125000 行；pursuit 每行还含 24*4 的 raw distance、reward 和 validity 数组及大量 scalar diagnostics，默认 JSON separators 又带空格。

现在复用各自现有 `runtime_logging.console_interval_sampled_steps=20000`，不新增配置字段。
独立 `next_training_metrics_log` 从 20000 开始，sampled_steps>=threshold 时才生成/写 snapshot；while 推进越过全部已跨阈值。
没有 modulo 条件，不复用 console counter，不改变 sampled_steps 的增长方式。
24-env 情况前三个定期记录为 20016、40008、60000，threshold overshoot 始终小于 24。
3M synthetic sequence 最终 **150 行**，相比 125000 行减少 **99.88%**，满足 <=152 / >99.8% 的验收目标。

## 字段、final 与 resume

两个 runner 原有 snapshot record 字段保留。combat 的 R1-R4、事件、episode、outcome、损失和 optimization diagnostics 均保留；pursuit 的三组数组及所有 scalar diagnostics 均保留。
仅触发时构建 record，全部 training_metrics writer 使用 `json.dumps(record,separators=(",",":"))`，无 gzip。
snapshot 仍是所选实际 transition 的数据，不将它改成 interval-window 平均；episode 字段在所选 transition 没有 completed episode 时仍可为 null。

缓存最后真实 `result/rows`；final 时若 last_training_metrics_step 与实际 sampled_steps 不同，补写缓存数据，不执行额外 environment step、rollout 或 evaluation。
成功写入后才更新 last_training_metrics_step；同一步不会重复写。输出 sampled_steps 严格递增且唯一。
final 使用实际计数，沿用已有目标不能被 num_envs 整除时的原始采样行为，不修改 training target/计数来凑日志。
若无任何真实 transition 且缺少对应 final 记录，会明确报错，不伪造 final 数据或额外运行环境。

MAPPO resume 流式读取现有文件最后合法整数 sampled_steps，缺失文件为 None。
next threshold 使用 `(sampled_steps//console_interval+1)*console_interval`。
CLI 在既有 rollback 后再次刷新日志状态，避免缓存 rollback 前的 future row。
旧高频历史无需先 compact；无 rollback 时原历史 bytes 保持，后续低频追加。
稀疏 future rows 的 rollback 仍按 >checkpoint_step 删除并保留原有 backup 机制，不要求 checkpoint 恰好有日志行。
rollback 重新写 training_metrics 时同样 compact；optimization/evaluation 的原 rollback 数据与格式逻辑保留。

MADSAC 确认原本也是逐 vector step 写入，因此使用相同阈值/final/compact 机制。
其现有协议 `formal_exact_resume_supported=False`、checkpoint 不包含 replay，原 trainer.resume 明确拒绝 exact resume；本轮没有添加会改变协议的 MADSAC 续训功能。
共用 logger 的 restore 可兼容旧历史，但不声称 MADSAC 的算法状态已支持 exact resume。

## 离线 compactor

新增 `tools/compact_training_metrics.py`，只实现工具，不自动处理 outputs 中已有实验日志。

- `--input` 必填；`--output` 可选，默认源目录下 training_metrics.compact.jsonl。
- `--interval-sampled-steps` 默认 20000。
- 逐行解析、只保留一个 pending final record，内存不随行数增长；单行/schema 的大小仍是必要内存。
- sampled_steps 必须存在、为非负整数；bool/float/非法 JSON/非递增或重复输入明确拒绝。
- 保留 first、每个绝对 interval threshold 被跨越后的第一条、final；输出 unique/increasing，compact JSONL。
- 默认不覆盖原文件，也拒绝已存在 output 或 input/output 同路径。
- 只有显式 `--replace` 才写临时文件、flush/fsync/close、原文件改名为唯一 backup、atomic replace；安装失败时恢复原文件。
- scan 前后检查 size、mtime_ns 和文件 identity；发现变化拒绝 replace 并清理 temp。
- 打印 input/output rows、bytes、compression_ratio（input bytes/output bytes）、first/last sampled_steps、backup/output 路径。

1000 行不规则计数 synthetic 和 replace/active-growth 拒绝专项通过。
CLI smoke 只读取新建的 offline_synthetic.jsonl，写新的 offline_cli.jsonl；没有 compact、改写或删除任何已有训练 run 的 metrics。

## 日志与训练数学冻结

configs/mappo.yaml、configs/madsac_single_round_3m.yaml、configs/single_round_environment.yaml、configs/pursuit_environment.yaml 均未改；所有 env 源码、MAPPO/MADSAC network/trainer/replay/evaluation/checkpoint/protocol 的冻结文件保持一致，共核对 **34 个文件**。
environment_version/variant、MAPPO_IMPL_VERSION 和 checkpoint identity 没有升级或改变。
optimization_metrics 的 writer/cadence、evaluation_history、console/train.log、best selection、checkpoint 保存字段和评估方法保持原样。
日志状态不写入 checkpoint，不进入 reward、rollout、GAE、optimizer、RNG 或 policy selection。

短 CUDA A/B 使用相同 seed=73、初始网络、环境/算法配置、64 sampled transitions，A 通过 hook 恢复原每-step writer 调用频率，B 使用新默认稀疏 logger。
分别验证 MAPPO combat、MAPPO pursuit、MADSAC：

- actor、critic、MADSAC targets 及全部 optimizer state **torch.equal / 精确一致**。
- sampled_steps、update counters、episode count、training seed 集合及 evaluation result 精确一致。
- optimization_metrics.jsonl **bytes 完全一致**。
- MADSAC 有效 replay 内容及 replay/policy/Torch/NumPy/Python RNG 状态一致。

首次 MADSAC 额外 replay 验证误比较了 np.empty 的未使用槽位；修正为有效 size 区域及采样 RNG 后，完整测试通过。未修改 replay 实现或采样规则。
A 的 64 个 step logs 与 B 的一条 final row 是预期持久化差异。

真实 MAPPO resume smoke：先高频记录并训练 128 transitions，再用同配置 checkpoint 续到 256；保留前 128 行，追加稀疏记录 200 和 final 256，无重复。
该 smoke 仅将 logger 对象 interval 加速到 100，未更改任何 YAML 或训练参数；正式 interval 始终 20000。

## 验证结果与大小

WSL 返回 E_ACCESSDENIED，按 runtime policy 使用 PowerShell Windows uav：`D:\conda_envs\envs_dirs\uav\python.exe`。
CUDA GPU：NVIDIA GeForce RTX 5060 Laptop GPU，无 CPU training/checkpoint fallback。

- **Full pytest：499 passed in 174.21s**，含新增 20 项 logging 专项及三个 CUDA A/B、resume smoke。
- MAPPO combat CUDA smoke：128 transitions，1 条 final training_metrics，checkpoint CUDA 求值通过。
- MAPPO pursuit CUDA smoke：128 transitions，1 条 final training_metrics，checkpoint CUDA 求值通过，pursuit 数组字段保留。
- MADSAC CUDA smoke：128 transitions，1 条 final training_metrics，原 stochastic evaluation 与 checkpoint 保存通过。
- offline compactor function/CLI smoke、replace backup、增长中 replace 拒绝、稀疏 rollback、final unique、schema/compact encoding 检查通过。
- 原架构审计 16 frozen files / 4 frozen combat methods、日志专项 34-file 冻结审计和 git diff --check 通过。

真实 pursuit schema、24-env、3M 等价 **synthetic counter sequence** 生成：

| 指标 | 结果 |
| --- | ---: |
| 旧理论 rows | 125000 |
| 新实际 rows | 150 |
| rows 降幅 | 99.88% |
| 新实际 bytes | **794347**，约 **0.76 MiB** |
| 相同代表性 schema 的旧格式 bytes 估算 | 705328707，约 705 MB |

旧 bytes 是固定代表性 schema 的估算，不是压缩用户既有文件的测量；此 synthetic 只推动日志 counter，**没有执行 3M training**。

证据：reports/logging_pytest_final.txt、reports/logging_cuda_smoke.txt、outputs/logging_acceptance/evidence.json、outputs/logging_acceptance/logging_only_audit.json、outputs/logging_acceptance/architecture_audit/single_round_audit.json。
CUDA/synthetic/offline artifacts 全部位于新建 logging_acceptance 目录或 pytest 临时目录。

## 文件清单与范围

修改：algorithm/mappo/runner.py、algorithm/madsac/runner.py、algorithm/train_mappo.py、README.md。
新增：algorithm/common/training_metrics_logging.py、tools/compact_training_metrics.py、tools/validate_training_metrics_logging.py、tests/test_training_metrics_logging.py、本报告及两份验证日志。

未启动正式 500k/3M、pursuit 长训练；未对已有训练目录自动 compact；未提交、推送或创建分支。
