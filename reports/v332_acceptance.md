# v3.32 Red-only direct overload control 验收

日期：2026-10-14。冻结基线：`9f6dc576367a071c2b784ee1ada37ea822000e11`。
本轮新增版本隔离的 CombatEnvironmentV332，未覆盖默认环境或 v3.31。

## 单列的三条验收不变量

**A. Actor [0,0,0] target == [mu=0,xi_hat=0,xi_bar=1]。**

**B. Red executed ControlCommand == [nx=xi_hat,nz=xi_bar,phi=mu]，不重复乘 g。**

**C. Blue 仍使用原 heading/pitch/speed controller，没有进入 Red direct-control path。**

三条均由专项单元测试、源码冻结检查和实际环境/训练 smoke 验证。

## 1. 文件清单与版本入口

新增：

- configs/combat_environment_v332.yaml：v3.31 配置只改变 version、增加 Red 专用严格配置。
- configs/mappo_4v4_v332.yaml：只把 observation_dim=45 改为47，其余逐项与 v3.31 相同。
- env/v332_config.py：严格 schema、有限数/类型/精确参数、旧 v3.31 全配置冻结验证。
- env/v332_control.py：decoder、circular rate limiter、ControlCommand/observation 映射、只读诊断及动作标签适配。
- env/v332_observation.py：47D self9+ally18+enemy20。
- env/v332_transition.py：隔离的 transition core，只有 Red advance 调用不同。
- env/combat_v332.py：Red 独立状态和 advance、reset/observation/info，继承完整 Blue 控制链和战斗逻辑。
- tests/test_combat_v332.py：49 项专项测试。
- tools/audit_combat_v332.py：冻结、静态检查、真实环境和 bounded CUDA smoke。
- docs/v332_red_direct_control.md、本报告及验证日志。

修改：env/config.py 注册/validator/dimensions；env/combat_env.py constructor dispatch；algorithm/mappo/runner.py 的新版本诊断汇总和标签；algorithm/common/evaluator.py 的新版本报表兼容/标签；README.md 新版本索引。
没有修改 v3.31/v3.30 environment、旧 observation/transition、Blue policy/controller、weapon/reward/sensor、任何算法 network/trainer 或旧 YAML。

正式入口识别 environment_version='3.32'、dimensions=(47,3,4)，factory 继续通过现有 MultiUAVCombatEnv dispatcher 创建新类。
直接用 CombatEnvironmentV332 装载旧3.31配置也会拒绝，防止类/配置混用。

## 2. Red 数据流与精确映射

`normalized [a_mu,a_t,a_n] -> physical target [mu*,xi_hat*,xi_bar*] -> per-aircraft limiter -> executed [mu,xi_hat,xi_bar] -> ControlCommand -> existing PointMassDynamics/RK4`。

归一化输入先 clip 到 [-1,1]，拒绝错误 shape/非有限数：

| 维度 | 目标公式 | 物理范围 |
| --- | --- | --- |
| a_mu | mu*=pi*a_mu | [-pi,pi] rad |
| a_t | xi_hat*=a_t if a_t<0 else 2*a_t | [-1,2] |
| a_n | xi_bar*=1+a_n if a_n<0 else 1+7*a_n | [0,8] |

Actor zero action 严格对应目标 [0,0,1]；正端点 [pi,2,8]、负端点 [-pi,-1,0]。
两个 overload 无量纲，直接传入 nx/nz，不乘9.81。
Red 没有 heading/pitch/speed target、2s tracking controller、inverse dynamics 或旧 normal-load projection。
保留原 tanh Gaussian 3D policy；未改 PPO/GAE/logprob/Jacobian/entropy/optimizer。

## 3. Rate limiter、状态和 reset

每实例独立 float64 `red_control_states.shape=(4,3)`，顺序固定 [mu,xi_hat,xi_bar]；reset 所有行严格 [0,0,1]。
load 执行值先 clip 到 previous±rate*dt，再 clip 到对应物理范围。
roll error=wrap_angle(mu_target-mu_previous)，delta=clip(error,±(pi/2)*dt)，executed=wrap_angle(previous+delta)。
恰好相差180°时两方向等长，保留 raw requested difference 的符号，满足附件要求的正/负首步方向。

dt=.1 时每步最大变化为 [pi/20,.2,.2]。
从 trim 到全正端点首步 [pi/20,.2,1.2]；到全负端点首步 [-pi/20,-.2,.8]。
179° -> -179° 走 +2°，反向走 -2°，均不产生358°旋转。
roll 速度应通过 circular difference 衡量，跨 ±pi seam 的坐标跳变不是物理跳变。

只对 alive Red 更新 limiter 和 dynamics；dead action 不改变其控制状态或 aircraft state。死亡观测仍整行零。
并行 worker 各自拥有独立数组；真实双 worker 测试给一个环境全正动作、另一个 zero action，控制特征互不污染；使用新 episode seeds reset 后都回到 trim。
zero action 是 trim 目标，从非trim状态切换时仍需逐步限速返回，不能绕过 limiter。

## 4. Observation 45D ->47D

| 零基区间 | 布局 |
| --- | --- |
| 0..8 | self9: x,y,altitude,speed,mu,xi_hat,xi_bar,psi,theta 的规范化值 |
| 9..26 | 3 allies *6，原样平移 |
| 27..46 | 4 enemies *5，原样平移 |

control features：mu/pi；xi_hat<0 时 xi_hat、其它 xi_hat/2；xi_bar<1 时 xi_bar-1、其它 (xi_bar-1)/7。
trim features 为 [0,0,0]；±pi、xi_hat=-1/2、xi_bar=0/8 各端点规范化正确。
实现复用冻结45D builder，self 原4个状态和 psi/theta 数值不变，后38个 entity features 逐值复制。
敌机 weapon-consistent signed geometry 和 dead entity zero-fill 原样保留，没有重写 sensor/geometry。
MAPPO 新配置仅输入维数47，action_dim=3、num_agents=4；隐藏层、critic 类型、超参数不变，输入层尺寸变化由新 observation 自然决定。

## 5. Blue 与战斗冻结证据

CombatEnvironmentV332 **没有 override _advance**。
Red 调用 `_advance_red_direct`；Blue 调用继承的 v3.29 `_advance`，仍使用 action_to_control_v329 和原 action.command/controller。
Blue desired speed=225 m/s、三个2s响应常数、normal_load_max=10 保留；Red [0,8] 独立配置，拒绝10g漂移。
在同一 Blue state、同一显式 action 下，两版本 ControlCommand 和积分后状态逐值一致；实际完整 step 中旧 controller 只被4架 Blue 调用，Red 不调用它。

冻结审计核对 **346 个已有 env/config/algorithm 文件** 与 Git baseline 一致，唯一排除的是本轮必须扩展的四个 dispatcher/reporting 文件。
包括 dynamics/RK4、Blue policy/controller、weapon/fire/reward/sensor/scenario、全部旧配置、MAPPO/PPO/GAE/network、MADSAC/MAAC/STEA 等。
新 transition core 与旧 core 的 AST 只差 Red advance 的调用名；完整 step reward/info wrapper AST 完全一致。

额外做 seed=71 的 zero-action 整局回归，两版本在 **156 步**同时结束为 draw_mutual_destruction：
Red/Blue aircraft states、逐步 reward、termination/truncation、双方 ammo、RNG state 和平移后的 entity observation blocks 全程逐值一致。
这是相同 trim 输入下的协议回归，不声称一般动作/Red轨迹改变后 Blue 的战术响应轨迹也相同。

**当前 v3.31 实际 weapon range=1000 m、ammo=6、cooldown=1s，全部原样保留。**
本轮未把历史分支的4000m或论文的2km移入配置；speed=150..300、pitch=±pi/3、dt=.1、max_steps=1000、arena/ground/ceiling、初始化和 reward 均不变。
稳定连续火窗25步实测 Red attempts=[1,11,21]，hit 被设为 miss 仅用于该 cadence fixture，真实环境 eligibility/ammo/cooldown/transition 完整执行。

## 6. Diagnostics 和稀疏日志

info 提供 mean_abs_mu、mean_xi_hat、mean_xi_bar，以及三种 rate-limit fractions 和五种 near-limit fractions。
仅按本次实际执行的 alive Red 计数；dead 不计分母。near band 为 load 全范围1%、roll abs>=.99*pi。
没有加入 reward。
沿用当前 runner 的 rollout_summary，每 PPO update 一条；按 alive-agent 权重聚合标量，不写逐step大数组。
新版本仅将通用 trainer 指标中的 heading/pitch/speed 或 psi/theta/v 标签转换为 a_mu/a_t/a_n，保持数值和训练数学不变；Blue 原标签保留。
trainer 本体未改，旧版本日志标签不改。

## 7. Checkpoint 隔离

environment version、47D dimensions 和 config fingerprints 自然隔离：

- v3.32 environment SHA256：`8dfe1c9d5d41465eba29c566dd0e168c7c01976161109d476304944065b92a86`。
- v3.32 MAPPO config SHA256：`79cd961586e3b0b5b1caf3e445ebc2961ef7f739d7b3cb2ff19ec8c34e8145da`。

新 checkpoint 通过同版本 resume/evaluation 校验和实际 CUDA evaluation。
另外使用冻结 v3.31 的45D网络/factory与原 checkpoint saver 生成完整 old-protocol fixture（0 training steps），它通过旧协议校验，但被3.32 resume/evaluation以 environment_version mismatch 明确拒绝。
未声称该 fixture 是旧训练产物，也没有伪造旧训练结果或执行 actor迁移。
没有改 MAPPO_IMPL_VERSION；未修改 checkpoint 身份校验规则以绕过旧协议。

## 8. 测试、smoke 与未解决问题

WSL 返回 E_ACCESSDENIED，使用 AGENTS 指定 fallback：Windows PowerShell uav Conda，Python=`D:\conda_envs\envs_dirs\uav\python.exe`。
CUDA：NVIDIA GeForce RTX 5060 Laptop GPU，未进行CPU training fallback。

**v3.32 专项：49 passed in 2.94s。**
覆盖 decoder/clip/invalid input、精确首步、roll seam、300步随机rate/range、解析动力学/trim/no-double-g、严格字段、Blue隔离、完整step调用隔离、47D/dead slots、reset、真实vector隔离、旧checkpoint拒绝、冻结AST、物理动作标签。
静态 AST parse 和5个新环境模块 compile 检查通过，git diff --check 通过。

**MAPPO CUDA smoke：512 sampled transitions、2个真实spawn worker、64次PPO update、47D obs/3D actor、mlp critic，通过。**
采用原有 smoke 模式 hidden64/rollout4/2epochs；正式配置层宽256和全部超参数不变。
完整使用 Red decoder+limiter、原 dynamics/RK4、原 Blue、weapon/reward/termination；optimization metrics 有限，checkpoint 保存、校验及独立留出 CUDA evaluation 通过。
training_metrics 共64条 rollout_summary，包含新控制诊断；没有恢复逐vector-step JSON。
短训练本身无 completed episode，留出 evaluation 执行完整对局。smoke 只证明训练/协议通路，不代表性能结论。

### Full pytest 尚非全绿，32项全部为基线既有失败

最终完整执行：**2900 passed、32 failed、2 skipped in 265.56s**。
用 `git archive` 导出未修改的 `9f6dc57` 到本地 tmp，并在同一个 uav 环境重跑这32个失败 node IDs：**32个全部复现失败**。
最终失败 node ID 集合与 untouched baseline 集合完全一致，差集为空，没有新增测试失败。
未通过修改历史golden、安装额外依赖、伪造checkpoint、跳过测试或改旧环境来掩盖失败。

| 基线既有问题 | 数量 |
| --- | ---: |
| 旧配置目录清单过期 | 1 |
| MP4 writer/plugin 的 .mp4 错误 | 1 |
| 历史轨迹/完整info字节哈希不匹配 | 14 |
| 历史输出checkpoint缺失 | 2 |
| 旧日志测试对象缺少 env_config 属性 | 10 |
| 旧5v5 fixture严格tensor equality失败 | 4 |

以上原因在当前基线可复现，具体哈希/fixture偏差根因本轮未展开修复，以遵守冻结旧版本和算法实现的边界。
**未解决项：这32项历史全量测试失败仍存在；v3.32新增专项及指定smoke没有未通过项。**

证据：

- reports/v332_pytest_final.txt、v332_special_tests.txt、v332_baseline_failures.txt、v332_cuda_smoke_final.txt。
- outputs/v332_acceptance_final/evidence.json：冻结、compile、环境cadence、CUDA、checkpoint、diagnostics。
- outputs/v332_acceptance_final/pytest_comparison.json：完整失败集合及空差集。
- outputs/v332_acceptance_final/zero_action_331_332_regression.json：156步逐值回归。
- outputs/v332_acceptance_final/complete_v331_checkpoint_rejection.json：完整45D旧协议拒绝。
- outputs/v332_acceptance_final/mappo_cuda_smoke：最新checkpoint、配置快照、rollout/optimization JSONL 和 summary。

未启动500k、3M或其它正式长训练；未提交、推送或创建分支。
