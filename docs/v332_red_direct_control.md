# v3.32 Red-only physical overload control

v3.32 是 v3.31 的隔离版本，只改变 Red 动作/执行链及暴露 executed control 所需的 observation。默认环境和 v3.31 文件不变。
配置：`configs/combat_environment_v332.yaml`、`configs/mappo_4v4_v332.yaml`。
动作公式以本轮用户附件明确给定的林安洋论文控制协议为实现依据，不额外引入论文其它环境参数。

## 三条验收不变量

A. Actor `[0,0,0]` 的目标严格为 `[mu=0,xi_hat=0,xi_bar=1]`。

B. Red executed control 直接构造 `ControlCommand(nx=xi_hat,nz=xi_bar,phi=mu)`，不再乘 g。

C. Blue 保持原 heading/pitch/speed controller，绝不进入 Red direct-control path。

## 动作与限速

Actor 仍为 tanh Gaussian 3D policy，三个归一化输出命名为 `[a_mu,a_t,a_n]`，先 clip 到 [-1,1]：

- `mu_target = pi*a_mu`。
- `xi_hat_target = a_t if a_t<0 else 2*a_t`。
- `xi_bar_target = 1+a_n if a_n<0 else 1+7*a_n`。

物理范围分别为 [-pi,pi] rad、[-1,2]、[0,8]；两个 overload 均无量纲。
每架 Red 独立保存 `red_control_states[i]=[mu,xi_hat,xi_bar]`，float64 shape (4,3)，reset 为 [0,0,1]。
仅 alive Red 更新状态、执行 dynamics；dead Red 保留死亡前 control，不消费 dead action。

`dt=.1` 下，roll 每步最多 pi/20 rad，tangential/normal load 每步最多 .2。
两个 load 先将 target clip 到 previous±rate*dt，再保证物理范围。
roll 使用 `wrap_angle(target-previous)` 的最短圆周误差，clip 到 ±roll_rate*dt，再 wrap executed roll 到 [-pi,pi)。
恰好相差 pi 时存在两条等长路径，保留 raw requested difference 的符号，使从 trim 到 +pi/-pi 首步分别为 +pi/20/-pi/20。
179° -> -179° 走 +2°，反方向走 -2°；roll 的变化率用 circular difference 衡量，不能用跨 seam 的普通相减。
pi/-pi 是等价方位，executed 在 seam 上的符号变化不是 2*pi 的物理旋转。

zero action 表示 trim **目标**。从 reset trim 开始持续 zero action 严格平飞；从其它 executed state 切回 zero action 仍按相同 rate limit 逐步返回 trim，不会瞬时跳变。

## 两条分开的执行路径

Red：normalized actor action -> v332 decoder -> per-aircraft limiter -> ControlCommand -> 原 RK4 / PointMassDynamics。

Blue：原 fixed policy / 显式 heading-pitch-speed action -> inherited `_advance` -> `action_to_control_v329` -> 原 controller -> 原 RK4 / PointMassDynamics。

CombatEnvironmentV332 不 override `_advance`；隔离的 transition core 只将 Red 调用改为 `_advance_red_direct`。
old v3.30 transition core 和 v3.31 reward/info wrapper 保持不写入；新 core 与旧 core 的 AST 除这一调用外完全一致，step wrapper AST 完全一致。
Blue controller 的 2 s time constants、10g cap 和 desired speed 225 m/s 全部保留；它的 10g 不用作 Red limit。

## 47D observation

从冻结的 45D builder 获取原有特征，仅替换 self block 并平移 entity blocks：

| 零基 offset | 特征 |
| --- | --- |
| 0..3 | x_norm,y_norm,altitude_norm,speed_norm，原样 |
| 4..6 | executed mu_norm,xi_hat_norm,xi_bar_norm |
| 7..8 | psi_norm,theta_norm，原样 |
| 9..26 | 三个 allies，每个 6D，原样 |
| 27..46 | 四个 enemies，每个 5D，原样 |

mu_norm=mu/pi。
xi_hat_norm=xi_hat（负值）或 xi_hat/2（非负）。
xi_bar_norm=xi_bar-1（小于1）或 (xi_bar-1)/7（其它）。
三个 trim control features 都为 0；extrema 分别归一化为 ±1。
dead focal agent 整行零；dead ally/enemy slot 严格零填充。
enemy distance/speed/aspect/weapon-consistent signed boresight geometry 定义不变；未引入新的 sensor gating。

## Diagnostics 与持久化

info 中 `red_direct_control_diagnostics` 保存本次 alive Red agent_steps 和以下统计，也将标量平铺到 info：
mean_abs_mu、mean_xi_hat、mean_xi_bar；roll/tangential/normal_rate_limit_fraction；roll_near_limit_fraction；xi_hat/xi_bar_near_min/max_fraction。
rate-limit fraction 统计当前目标变化超出该步 rate budget 的比例（1e-12 数值容差）。
near-roll-limit 使用 abs(mu)>=.99*pi；near-load-limit 使用总物理范围 1% 的边缘带，即 xi_hat<=-.97 / >=1.97，xi_bar<=.08 / >=7.92。
均按实际执行前 alive Red 计数，dead rows 不进入分母。全部 diagnostics 不进入 reward。

MAPPO 沿用当前每次 PPO update 的 rollout_summary JSONL；按 live-agent 数加权汇总标量，不增加 per-step raw control arrays 或高频 JSON。
仅 v3.32 将通用 trainer 返回的旧动作轴诊断标签转换为 a_mu/a_t/a_n；Blue 的旧 high-level diagnostic labels 保留。
trainer 源码及 PPO/GAE/log_prob/tanh Jacobian/optimizer 没有修改。

## 版本与冻结范围

environment_version='3.32'，dimensions=(47,3,4)。validator 严格检查 Red 字段集合、有限数、类型和精确规定值；其余配置去掉新 block、映射回 version 3.31 后通过原 validator。
旧 Blue action.command/controller 继续保留，任何 Red normal_load_max=10 的复用均会被拒绝。
MAPPO 新配置仅将 observation_dim 从45改为47，所有其它网络设置和正式超参数逐项相同；没有改变 critic 类型或网络层宽。
checkpoint 原有 environment_version、dimensions 和 config SHA 区分两个版本；旧 3.31 完整 checkpoint 拒绝作为 3.32 resume/evaluation 加载。

实际 v3.31 基线的 weapon range 为 **1000 m**、每机 ammo=6、cooldown=1 s；这些值原样保留。
没有引入论文其它 speed=100..400、2km weapon、250步 horizon 或连续锁定要求，也没有把历史其它分支的 range 搬进来。
保持当前 speed=150..300、pitch=±pi/3、dt=.1、max_steps=1000、arena/ceiling、4v4 initialization、sensor、battle reward、death/outcome/termination 全部不变。
MADSAC、MAAC、STEA-MAPPO 和旧配置不作修改。

验证工具为 `tools/audit_combat_v332.py`；可选 `--cuda-smoke` 只运行512 sampled transitions。
完整结果及既有测试问题见 `reports/v332_acceptance.md`。未启动任何正式长训练。
