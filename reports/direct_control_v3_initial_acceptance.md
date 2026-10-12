# single_round_v3 / pursuit_v1 验收报告

日期：2026-10-12。冻结基线：`81d623e1ecafbf7df91aa2d9dc1fc19cf885f2a9`。
本轮只实施 direct physical action、对应固定 Blue guidance、独立 pursuit curriculum，以及任务身份和 actor 初始化接口；未启动正式长训练。

## 版本、动作与执行路径

正式 combat 为 `environment_version: '2.4'`、`environment_variant: single_round_v3`、`task_type: combat`、`action_semantics: direct_physical_control`。
pursuit 为同一版本、`environment_variant: pursuit_v1`、`task_type: pursuit`、同一 action semantics。
v2 配置及关键源码保存在 `legacy/single_round_v2`，快照内容逐一对照上述不可变 Git 基线；v2 不再是正式入口。

Actor 动作维度仍为 `(4,3)`，依次是归一化 thrust、lift/normal overload、roll。

| 归一化动作 | 物理量 | 范围 |
| --- | --- | --- |
| a_T | T = 2 a_T | [-2,2] |
| a_L | L = 5 a_L | [-5,5] |
| a_beta | beta = (pi/3) a_beta | [-pi/3,pi/3] rad |

解码先将输入 clip 到 [-1,1]，然后直接返回 `ControlCommand(nx=T,nz=L,phi=beta)`。
正式 action 配置仅包含 mode 和三组物理范围，旧 command/controller 参数已退出正式配置。
Red `_advance` 只调用 `normalized_action_to_direct_control`，继而原样调用 RK4/PointMassDynamics。
没有 ManeuverTarget、时间常数、inverse dynamics、normal-load projection、平滑、执行器延迟或 L 偏置。
旧 `action_to_target` / `target_to_control` / `action_to_control` 定义保持完整，历史控制器测试使用冻结 v2 配置。

零动作产生 T=L=beta=0，并不 trim。在 theta=0、v=225 时：
`dv/dt=0`，`dtheta/dt=-9.81/225=-0.0436 rad/s`，`dpsi/dt=0`。
水平 trim 是物理 `[0,1,0]`，即归一化 `[0,0.2,0]`。
端点、clip、非有限输入及解析导数测试使用严格绝对 tolerance；解析测试还覆盖正负 lift、roll、thrust 的方向。

## 冻结范围

`env/dynamics.py` 和 `env/integrator.py` 未写入修改。16 个冻结源码文件的 Git blob 与 v2 一致：

- env：dynamics、integrator、weapon、reward、observation、scenario、fixed_policy、models、geometry、math_utils。
- algorithm：common/vector_env；MAPPO networks/trainer；MADSAC networks/trainer/evaluation。

文件审计将 CRLF/LF 按 Git 文本规则规范化后比较 blob，且这些冻结文件无工作区 diff。
额外 AST 比较确认 combat 的 `step`、`_attack_attempts`、`_resolve_combat`、`_resolve_noncombat_losses` 方法体与 v2 完全一致。
正式配置除版本/variant/task/action identity、action 和 Blue guidance 外，不得变化；两个正式算法配置也与基线一致。

保留 v=150..300、theta=±pi/3、dt=.1、horizon=1000、arena=5000、4v4 对向初始化、52D 特征定义和全部 observation scales。
已有 previous executed phi slot 现在自然记录实际 beta，未增维；专项测试核对 `(4,52)` 和实际执行值。
战斗奖励仍为 R1+R2+R3+R4：kill +10、own loss -10、boundary -10、approach .001、advantage/threat tiers 原值。
未添加任何飞行、控制努力或论文中的替代战斗奖励。
weapon geometry、概率模型、4000 m range、1 s cooldown、next_fire_step、双方先尝试后同时结算全部保持。
真实连续火窗 25 步实测 attempts `[1,11,21]`，3 次尝试、22 个 cooldown-blocked agent steps。

## Blue direct guidance 与地面保护

新 `DirectPursuitPolicy` 确定性选择最近 alive Red；同距以 index 决定，desired_speed 保持 250 m/s。
速度请求为 `(250-v)/speed_response_time_seconds`，thrust 为 `sin(theta)+requested_acceleration/g`，clip 到 [-2,2]。
heading/pitch error 除以对应 response time 得到请求角速度，并分别限幅。
仅 Blue 反算：vertical = cos(theta)+v*pitch_rate/g；lateral = v*cos(theta)*heading_rate/g。
beta 由 atan2(lateral,vertical) 得到并限幅；L 是该合法 beta 方向上的所需负载投影，再限幅到 [-5,5]。
限幅时不保证精确实现原请求角速度，但始终输出合法物理动作。
Blue 最终输出归一化 `[T/2,L/5,beta/(pi/3)]`，由与 Red 同一个 decoder 执行。

所有 guidance 参数都在 blue_policy：三个响应时间默认各 2 s，heading rate bound .3 rad/s、pitch rate bound .15 rad/s。
gravity 直接复用 PointMassDynamics 的固定值；物理控制限制与正式 decoder 一致。
旧类仅复用四槽诊断计数逻辑，旧 delta action 和 ground-risk/controller 实现均被覆盖，不执行。

地面保护用显式 `recovery_time_seconds=4`，对应旧 2*2 s 时间尺度。
`downward=max(-v*sin(theta),0)`；超过 epsilon=1e-6 时，altitude/downward <=4 s 触发恢复。
边界只使用一个 ULP 的向上容差处理三角函数/除法舍入。
恢复将 beta=0、L=`recovery_lift=5`，T 继续保持合法。
专项从 altitude=450 m、v=225、theta=-pi/6 开始运行 100 步：最低高度 **243.9659 m**，未撞地。
另有最近 alive 目标、确定性、合法范围和 reset 诊断测试。该结果是简单俯冲专项，不代表所有极端状态均可救回。

## 独立 pursuit 任务

`PursuitEnv` 是单独环境类和配置，不在 combat 中切换 reward。
只复用 integration/observation helper、同一 AircraftState/spec、direct decoder、52D foundation 和四-agent MAPPO 架构。
不存在 weapon 对象、attack 配置、fire clock、攻击或 weapon kill 路径。
初始位置在半径 3000 m 圆盘内随机，altitude 2000..4000 m、speed 200..250 m/s、pitch ±.1 rad、heading ±pi。
对手每 20 步从显式配置的归一化 T/L/beta 区间采样一次并保持；只使用环境 episode RNG，可由 seed 完整复现。
地面/边界损失仍使对应 UAV 失活；一方无 alive UAV 时结束，否则 1000 步截断。
**1000 步 horizon 是 project implementation choice，不声称为论文原值。**

唯一奖励为 `r_i = -raw_nearest_distance_m[i] * distance_reward_scale`，默认 scale=1.0，无隐式归一化。
距离针对最近 alive opponent；沿用既有 alive-mask 架构，失活 UAV 返回 0，并显式记录 validity mask，避免将其误读成零距离追踪成功。
本轮没有增加存活奖励、坠机惩罚、速度/高度/姿态/队形奖励。

info/training JSONL 记录逐 agent raw_nearest_distance_m、pursuit_reward、validity mask，以及：

- mean/final/minimum nearest distance；Red 与 opponent 的 ground/boundary loss count。
- mean speed、speed min/max 与 pitch min/max 的 envelope hit ratio。
- mean_abs_T/L/beta 与 fraction_abs_T/L/beta_near_limit，near-limit 阈值默认为物理限幅的 95%。

mean/min distance 使用有效 alive-agent 样本；final distance 使用各 agent 最后有效距离的平均，防止全灭后得到虚假的零距离。
速度/控制统计对积分前 alive 的 Red 样本计数。全部 diagnostics 不影响 reward，专项对比不同 diagnostic threshold 的奖励逐项一致。
pursuit 求值和 best selection 独立于 combat outcome：以 average team return 排序，不伪造 win/kill/R1-R4。

## Checkpoint 与 actor-only 迁移

MAPPO/MADSAC 新 checkpoint 均写入 version、variant、task_type、action_semantics；resume/evaluation 校验匹配及原有 config hash/runtime identity。
v1/v2 的 2.3/delta-command 身份明确拒绝；正常 pursuit -> combat resume/evaluation 明确拒绝。
MAPPO 新参数 `--init-actor-checkpoint` 仅适用于 fresh combat v3，和 resume 同时指定时在创建 run 前拒绝。
source 必须为兼容 MAPPO implementation、2.4/pursuit_v1/pursuit/direct_physical_control、52/3/4，并匹配 actor 参数 key/shape，参数必须有限。
加载只执行 actor.load_state_dict；未加载 critic、optimizer、sampled_steps、update counters 或 evaluation/episode history。
run_config 和后续 checkpoint 记录 source 完整路径、SHA-256、source task/action semantics、actor_only 标记。
64-hidden smoke actor 与 256-hidden 正式 actor 因架构不同不能互载；本轮保留原正式网络/训练器和超参数。

逐参数专项及 smoke 初始化前验证：source actor 精确复制，destination critic 完全不变，两个 optimizer state 为空、sampled_steps=0、evaluation history 为空。
MADSAC 仍为正式 combat baseline；本轮 pursuit 和初始化接口使用 MAPPO。

## 验证结果

WSL 查询返回 E_ACCESSDENIED，因此按照 runtime fallback 使用 PowerShell Windows uav Conda 环境。
Python：`D:\conda_envs\envs_dirs\uav\python.exe`；CUDA GPU：NVIDIA GeForce RTX 5060 Laptop GPU。

- 完整 pytest：**458 passed in 35.77s**。保留原 420 项回归，新增 38 项 direct/pursuit/迁移测试；历史 controller 测试改读 v2 快照。
- 解析 decoder/dynamics、Blue guidance/recovery、cadence、checkpoint incompatibility、正常 pursuit 同任务 resume、缺失 action/task identity、CLI init/resume 冲突检查通过。
- 结构审计：16 frozen files、4 frozen combat methods、Red legacy-controller 排除、配置冻结、435 个历史 archive 和 9 个 v1/6v6 snapshot 检查通过；v2 快照逐项验证。
- clean-clone 测试检验当前**已提交 HEAD**，不复制本轮未提交 patch、不创建临时 commit；本轮工作区实现由 full tests 和工作区审计验证。

| CUDA smoke | transitions | PPO updates | completed training episodes | checkpoint evaluation |
| --- | ---: | ---: | ---: | --- |
| combat v3 | 1200 | 19 | 9 | passed |
| pursuit v1 | 1200 | 19 | 4 | passed |
| pursuit actor -> fresh combat | 1200 | 19 | 9 | passed |

以上均使用既有 smoke 架构 hidden=64、2 PPO epochs；smoke tool 将 rollout 调为 64 提高验收速度。正式配置完全未改。
每组均有有限 optimization metrics、唯一 reset seeds、实际 checkpoint 保存及 CUDA 严格求值。
actor 初始化 smoke 初次验证断言因 CPU/CUDA tensor 比较报错，修正验证断言的设备对齐后仅补跑该组；最终 evidence 全部通过。
1200-transition smoke 用于协议/数值通路验证，不代表已学会追踪或战斗。pursuit 留出求值仍有 2 ground losses 和 2 boundary losses，没有隐藏这些诊断或将其加入 reward。

证据文件：`reports/v3_pytest_final.txt`、`reports/v3_cuda_smoke.txt`、`outputs/direct_control_acceptance/evidence.json`、`outputs/direct_control_audit/single_round_audit.json`、`outputs/cadence_v3_validation/cadence.json`。
每组 latest.pt、env_config.yaml、algorithm_config.yaml、training/optimization_metrics 和 run_summary 留在对应 smoke 输出目录。

## 修改文件清单

- 配置/环境：configs/single_round_environment.yaml；新增 configs/pursuit_environment.yaml；env/config.py、control.py、combat_env.py、factory.py；新增 direct_protocol.py、direct_policy.py、pursuit_env.py。
- 协议/训练求值：algorithm/common/protocol.py、checkpoint.py、evaluator.py；新增 pursuit_metrics.py；algorithm/mappo/runner.py、evaluation.py；新增 actor_initialization.py；algorithm/train_mappo.py；新增 train_pursuit.py；algorithm/madsac/protocol.py、runner.py。
- 验证：tools/audit_single_round.py、validate_attack_cadence.py、smoke_single_round.py；新增 smoke_direct_control.py；tests/test_combat_environment.py、test_fixed_four_runtime.py、test_single_round_protocol.py、test_training_protocol.py；新增 test_direct_control_v3.py。
- 文档/历史：README.md、本报告、最终验证日志、legacy/single_round_v2 下 9 个 v2 源码/配置/审计报告快照及索引。

未运行 500k、3M 或 pursuit 长训练；未提交、推送、创建分支或执行远程操作。正式训练命令等待用户确认后再提供。
