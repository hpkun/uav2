# single_round_v3 pre-training closure / pursuit_v2 验收

日期：2026-10-12。本轮基线：`e05130e4a20d81799762996f9ea14a61867c8339`。
上一轮初始验收报告保留为 `reports/direct_control_v3_initial_acceptance.md`；旧 pursuit_v1 配置和环境快照在 `legacy/pursuit_v1`，按上述 Git commit 验证内容。

## 本轮解决的问题

pursuit_v1 的 alive UAV 持续接受负距离奖励，但穿地/出界后失活、后续奖励归零，GAE 又随 alive mask 截断；提前死亡可能提高 return。这是 early-death reward hacking。
原 random raw T/L/beta 对手也会自己失活，移除有效目标；以米为 reward 的数值尺度过大；combat Blue ground guard 仅检查当前下俯，遗漏危险 commanded descent 的提前判断。

本轮分别修复这些问题，没有重新设计 direct decoder 或物理 guidance 核心，没有启动正式长训练。

## pursuit_v2 精确定义

正式配置身份为 `environment_version: '2.4'`、`environment_variant: pursuit_v2`、`task_type: pursuit`、`action_semantics: direct_physical_control`。
pursuit 是独立、无武器、固定 horizon 的 tracking curriculum。

- 四个 Red 和四个 opponent 全程 alive=True，两个 alive mask 全程 `[1,1,1,1]`。
- 穿地与出界只增加 diagnostics，不更改 state.alive，不抹除奖励，也不结束 episode。
- step 1..999：terminated=False、truncated=False。
- step 1000：terminated=False、truncated=True；之后必须 reset。
- 只有非法输入、非法失活状态或非有限数值等程序错误才抛异常，不能静默转为死亡或零奖励。

1000 steps = 100 s，仍是 **project implementation choice**，不声称为论文原值。MAPPO trainer 未改，通过环境语义保证 GAE 不再因 pursuit 失活截断。

### 唯一奖励与距离日志

`reward[i] = -nearest_opponent_distance_m[i] * 0.001`，等于负距离公里数。例如 2000 m -> **-2.0**。
配置明确写 `reward.distance_reward_scale: 0.001`；formal validator 严格拒绝其它 scale、其它 reward 字段及旧 pursuit_v1 身份。
没有生存奖励、死亡惩罚、ground/boundary penalty、姿态/速度/控制努力/队形奖励，也没有其它 normalization。

每一步四个距离均 finite 且 >=0，四个 opponent 始终是有效 target。
距离不再具有 dead-agent ->0 的分支；`nearest_distance_valid_mask` 始终四个 True。
info 和 training JSONL 同时保存 `raw_nearest_distance_m` 与 `pursuit_reward`，逐项满足上述缩放等式；返回 reward 的 float32 仅是原训练接口数值类型。

### Violation diagnostics

分别记录 Red 和 opponent 的 ground/boundary：

- `*_violation_steps`：违反约束的累计 agent-step 数，after-integration altitude<=0 或 horizontal radius>arena.radius。
- `*_violation_agents`：本 episode 至少违反一次的不同 aircraft 数，使用逐 aircraft OR mask 计数。
- `*_violation_rate`：violation_steps / (4*elapsed_steps)。

命名为 `red_ground_violation_*`、`red_boundary_violation_*`、`opponent_ground_violation_*`、`opponent_boundary_violation_*`。
旧 ground_loss_count/boundary_loss_count 等死亡语义字段已移除。
同一 aircraft 同一步同时穿地/出界时，两个诊断分别计数；均不改变 reward、alive 或 termination。

保留 mean/final/minimum nearest distance、mean speed、speed/pitch min/max hit ratios、mean_abs_T/L/beta 和三种 near-limit fractions。
mean distance 使用四 agent 的全部 step 样本，final distance 为最终四-agent 距离均值；minimum 仅用于诊断。
这些 metrics 进入 info、训练日志、求值和 summary，均不进入 reward。

## 随机 waypoint opponent

新 policy 为 `seeded_random_waypoint_direct`。每架 opponent 独立维护 `[x,y,altitude]` waypoint 和 desired speed。
所有采样只使用 pursuit episode RNG，同 seed 初始状态、完整 waypoint/speed 序列和完整轨迹逐值一致，不使用 global np.random。

| 参数 | 正式值及含义 |
| --- | --- |
| waypoint_interval_steps | 50；step index 0,50,...,950 各采样一次，每集 20 次 |
| waypoint_radius_m | 2500；r=R*sqrt(U)、angle uniform，圆盘面积均匀 |
| waypoint altitude | 2200..3800 m，uniform |
| desired speed | 200..250 m/s，uniform |
| speed/heading/pitch response time | 各 2 s，复用既有 direct guidance |
| heading/pitch rate bounds | .3 / .15 rad/s |
| ground recovery | 4 s；L=5、beta=0；epsilon=1e-6；考虑 commanded pitch |
| opponent-only boundary safety | radius>=3000 m 时暂时朝 arena 中心引导，目标 altitude=3000 m、speed=200 m/s |

流程为 waypoint -> desired heading/elevation/speed -> 与 combat Blue 完全相同的 `action_toward` 物理 guidance -> normalized T/L/beta -> 同一个 direct decoder -> 原 PointMassDynamics/RK4。
waypoint 50 步保持，但 normalized control 每步重新计算；无 teleport、直接 state 修改或第二套 dynamics。
边界 safety 仅作用于 pursuit opponent，不改 arena radius、不影响 Red、不进入 reward，也不改变 combat Blue 最近 alive Red 的战术规则。

### 100-seed safety 验证

`tools/validate_pursuit_waypoints.py` 对 seeds **0..99** 各完整执行 **1000 步**，Red 使用 normalized trim `[0,.2,0]`。
实际执行 **100000 simulation steps / 400000 opponent agent-steps**，每一步检查 fixed horizon、全 alive mask 和 reward finite。

| 结果 | 数值 |
| --- | ---: |
| opponent ground violation steps | **0** |
| opponent boundary violation steps | **0** |
| opponent ground violation agents | **0** |
| opponent boundary violation agents | **0** |
| 最低 opponent altitude | **2006.859888 m** |
| 最大 opponent horizontal radius | **4236.601250 m**，arena radius 5000 m |

全部 100 集无对手穿地/出界，不通过失活、重置、teleport 或截断掩盖违规。
逐 seed 证据保留在 `outputs/pursuit_v2_waypoint_validation.json`，进度和最终结果在 `reports/pursuit_v2_waypoint_validation.txt`。
这是规定的 100-seed 验证范围，不声称数学上覆盖所有可能种子。

## Best checkpoint selection

pursuit 正式 selection key 为 `(-mean_nearest_distance, -final_nearest_distance)`：mean 越小越好，仅完全相等时比较 final。
不以 average_return 或 minimum_nearest_distance 主导 best_eval.pt。average_return 和 average_agent_return 仍保留。
专项构造 return 更高但 mean distance 更差的 A，以及 return 更低但 mean 更好的 B，确认选择 B；另验证 final tie-break。
combat selection 完全不改：win rate、average return、fewer Red losses。

## Combat Blue commanded-descent guard

`ground_risk(own,commanded_pitch)` 恢复：
downward_current=max(-v*sin(current_theta),0)，downward_commanded=max(-v*sin(commanded_pitch),0)，downward=max(两者)。
downward<=epsilon 时无风险，否则以 altitude/downward<=4 s 判断；保留一个 ULP 的阈值舍入容差。
nearest alive Red 的 desired pitch 在危险下俯执行前参与判断，风险时 beta=0、L=recovery_lift=5，T 保持原 speed guidance 的合法范围。

专项覆盖 current descent、当前 theta=0 的 commanded descent anticipation、time-to-ground>4 s 的安全下俯、climb 不额外触发、原俯冲恢复场景；全部通过。
Blue `action_toward` 方法体与上轮 commit 的 AST 完全一致，仅 ground-risk/调用点及身份校验改变。

combat variant 保持 single_round_v3，新增 mandatory `blue_policy.ground_avoidance.consider_commanded_pitch: true`。
**v3 combat config fingerprint changed because commanded-descent ground guard semantics were corrected before formal training.**
原 SHA：`d89661900b71f281963d343cd1989d84da6ddc16d7cfcf11df1bf1bf33f3450c`。
新 SHA：`74dd228a741d6c1cfa974e015e3fd00a40ec945729da27c2066146314d184f58`。
审计确认 combat config 相比上轮仅新增此字段；无需升级 v4，旧 v3 checkpoint 不允许静默兼容。

## 冻结与 checkpoint identity

T=2*a_T、L=5*a_L、beta=pi/3*a_beta 的 decoder 文件与上轮 Git blob 完全一致。
16 个既有冻结源码与 v2 基线一致；combat step、attack attempts、simultaneous resolution 和 noncombat losses 的四个方法体保持一致。
PointMassDynamics、RK4、v=150..300、pitch=±pi/3、dt=.1、combat 4v4 初始化、52D 特征、4000 m weapon/hit model、R1–R4 均未修改。
1 s cooldown 实测 cadence 仍 `[1,11,21]`。MAPPO/MADSAC networks、trainers 和正式配置/超参数未改，MADSAC 代码本轮无修改。

新的 actor initialization 只接受 2.4/pursuit_v2/pursuit/direct_physical_control、52/3/4 和匹配 actor architecture；只加载 actor，不继承 critic/optimizer/counters/history。
正常 pursuit resume/evaluation 要求 pursuit_v2 identity 和精确 config hash。

使用实际上一轮 checkpoint 验证：

- `outputs/direct_control_acceptance/pursuit/latest.pt`：normal resume、evaluation、combat actor initialization 均因 pursuit_v1 variant 被拒绝。
- `outputs/direct_control_acceptance/combat/latest.pt`：resume 和 evaluation 均因旧 environment_config_sha256 被拒绝。
- v2 delta-controller/version 身份仍被拒绝；缺失或 false/non-boolean commanded-pitch flag 被 formal validator/环境拒绝。

这些实际拒绝原因记录在新 smoke evidence，而非通过修改旧 checkpoint 伪造迁移。

## 最终验证与 CUDA smoke

WSL 查询仍返回 E_ACCESSDENIED，按 AGENTS fallback 使用 Windows PowerShell uav 环境：`D:\conda_envs\envs_dirs\uav\python.exe`。
CUDA：NVIDIA GeForce RTX 5060 Laptop GPU；没有 CPU training/checkpoint fallback。

- Full pytest：**479 passed in 43.21s**。
- 新收口专项：**21 passed in 7.59s**，含 full-horizon 穿地/出界后持续负奖励、双方 alive mask、scale=-2.0、scale 拒绝、完整 1000 步同 seed 轨迹、waypoint 50-step 更新、无 teleport、best selection、guard 五类覆盖及 incompatibility。
- 100 seeds × 1000 steps：passed，对手所有 violation 为 0。
- cadence 与结构审计：passed；静态 AST parse 和 git diff --check passed。

| CUDA smoke | transitions | PPO updates | 完整训练 episodes | checkpoint CUDA evaluation |
| --- | ---: | ---: | ---: | --- |
| combat v3 | 1200 | 19 | 9 | passed |
| pursuit_v2 | 1200 | 19 | 1，**episode length=1000** | passed，求值 length=1000 |
| pursuit_v2 actor -> fresh combat v3 | 1200 | 19 | 8 | passed |

沿用 smoke hidden=64、2 PPO epochs、tool rollout=64，正式 architecture/hyperparameters 完全未改。
三组 optimization metrics 均有限，保存 checkpoint、唯一 reset seeds 及完整求值证据。
actor 初始化前逐参数确认 actor exact copy、critic unchanged、optimizer state empty、sampled_steps=0、evaluation history empty。

pursuit 留出 smoke 中 Red 仍有 3161 ground violation agent-steps、3001 boundary violation agent-steps，分别占 0.79025 / 0.75025；四 Red 都曾违规。
这些违规没有死亡、奖励抹零或提前结束；opponent 两类 violation 为 0，episode length 始终 1000。
smoke 是协议/数值通路验收，不代表 1200 transitions 已学会稳定追踪或安全飞行。

新证据路径：`reports/pursuit_v2_pytest_final.txt`、`reports/pursuit_v2_special_tests.txt`、`reports/pursuit_v2_cuda_smoke.txt`、`outputs/pursuit_v2_acceptance/evidence.json`、`outputs/pursuit_v2_audit/single_round_audit.json`、`outputs/cadence_v3_validation/cadence.json`。
clean-clone 测试仍只检验已提交 HEAD，不复制未提交 patch、不创建临时 commit；本轮工作区实现由 full tests 和 working-tree audit 验证。

## 修改文件清单

- 配置：configs/pursuit_environment.yaml；configs/single_round_environment.yaml（仅 ground-guard flag）。
- 环境：env/pursuit_env.py、direct_policy.py、factory.py；新增 waypoint_policy.py。
- 协议/日志/选择：algorithm/common/protocol.py、pursuit_metrics.py；algorithm/mappo/runner.py、actor_initialization.py。
- 验证：tests/test_direct_control_v3.py；新增 test_pursuit_v2_closure.py；tools/audit_single_round.py、smoke_direct_control.py；新增 validate_pursuit_waypoints.py。
- 文档/历史：README.md、本报告、direct_control_v3_initial_acceptance.md、legacy/pursuit_v1 的旧配置和环境快照、四份新验证日志。

没有启动 pursuit long training、500k 或 3M；没有提交、推送、创建分支或远程操作。
