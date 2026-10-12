# single_round_v2 最终科研实验收口验收

正式环境最终严格固定 **4v4 / 52D observation / 3D action**，version=2.3，variant=single_round_v2，
dt=0.1s，max_steps=1000。1.0s / 10-step per-aircraft cooldown 完整保留。
本轮移除误混入正式 runtime 的 6v6 泛化，恢复原 4v4 数值语义；不修改已经正确的 cadence。

## 修改文件与泛化移除

- `env/observation.py`：恢复不可变 pre-cadence 源码；固定基础 52D、固定输出 (4,52)，不根据 team_size 推导 78D。
- `env/scenario.py`：恢复 pre-cadence 源码，严格要求 team_size=4、四个 formation offsets。
- `env/fixed_policy.py`：恢复 pre-cadence 源码，去除 team_size 构造参数，三个内部诊断数组固定 (4,)。
- `env/combat_env.py`：保留 v2 cooldown；不再从 config 赋值 self.team_size；初始化明确检查 4/52，不向 GroundAware 传 team_size。
- `algorithm/common/protocol.py`：显式锁定 52/3/4，去除 13×team_size 表达，保留 v2、dt、时限和 attack 严格校验。
- `algorithm/mappo/runner.py`：完整恢复 pre-cadence 源码，维度依据固定环境常量，先执行公共 validator。
- `algorithm/madsac/protocol.py`：配置与 checkpoint 明确锁定 52/3/4，v2 身份、SHA、seed、50 episodes、44M/770001 全部保留。
- `algorithm/common/evaluator.py`、`algorithm/madsac/evaluation.py`：恢复 pre-cadence 源码，return vector/buffer 固定四个 agents。
- `algorithm/common/vector_env.py`：恢复 pre-cadence 源码；alive masks 固定 (num_envs,4)，没有动态重建 agent 数量的契约。
- `tools/audit_single_round.py`：19 文件直接比较 immutable pre-cadence commit，配置 exact delta，分离历史 archive provenance。
- `tests/test_formal_single_round_closure.py`：clean-clone regression 去除复制、整体 add、临时 commit；更新 baseline 比较测试。
- 新增 `tests/test_fixed_four_runtime.py`：九项固定 4v4/不可变来源回归。
- 删除 `reports/cadence_v2_baseline.json` 自声明冻结清单；更新 README 和本报告。

现有 legacy 文件没有修改；历史 6v6 文件继续隔离，不能进入正式 factory、训练或默认 pytest。
没有修改任何正式网络/trainer/超参数，没有改变 vector auto-reset/terminal transition/seed 算法。

## 不可变基线与 exact delta

正式 immediate pre-cadence baseline：
`89fd4c54d07205bd0952eb972cf133b4a41349f7`。
审计通过 git ls-tree / git show 从该已提交身份读取 blob/配置，不依赖当前 index 或工作区自声明 hash。

严格匹配的 **19 个文件**：

- env/control.py、dynamics.py、integrator.py、weapon.py、reward.py、observation.py、scenario.py、fixed_policy.py、models.py、geometry.py、math_utils.py。
- algorithm/common/evaluator.py、vector_env.py。
- algorithm/mappo/networks.py、trainer.py、runner.py。
- algorithm/madsac/networks.py、trainer.py、evaluation.py。

全部与 pre-cadence commit Git blob 等价（按 Git 文本 CRLF/LF 规范比较）。没有放宽 frozen list，
没有 TEAM_SIZE_REFACTORS 豁免。尤其 weapon.py、reward.py、observation.py、scenario.py、fixed_policy.py 全部一致。
Weapon docstring、true-3D gate、4000m 射程、hit probability/Gaussian sampling 未改。
R1–R4 数值、公式与结算顺序未改，不增加 cadence/fire/hit/safety/ammo reward。

环境配置也直接读取 PRE_CADENCE_COMMIT 中正式 4v4 配置，对照当前完整字典：
**只有 environment_variant 从 single_round_v1 改为 single_round_v2，以及新增 attack 段**。
simulation、action、aircraft、arena、scenario、weapon、reward、observation、blue_policy 完全一致。
MAPPO/MADSAC 算法配置完整字典也与 pre-cadence 配置一致。

历史 archive provenance 独立保留：
`8d09df1fd1e6d2425b712d0c336e9525cd313c4c`，435 个原始历史文件匹配该来源。
九个较新 v1/6v6 快照原本含上一轮未提交的泛化，因此不能伪称来自正式 4v4 baseline；
它们改为直接匹配已存在的 archive snapshot commit
`b80337fd7aad8a37865f10e3f2cb6451d2775b96` 中对应 legacy blob，内容全部保持不变。
这第三个身份仅用于历史快照溯源，不豁免任何正式 runtime 冻结校验。

## Clean committed clone regression

测试直接 local clone 当前 committed HEAD，不复制本轮未提交文件，不 add 整体 runtime，不创建临时提交。
clone 初始 clean；两次 structural audit JSON 完全相同；审计后 tracked working tree 仍 clean。
随后只在临时 clone 中故意修改/stage dynamics.py，immutable frozen comparison 明确拒绝。
单独执行结果：**1 passed in 7.58s**，日志 `reports/cadence_v2_clean_clone.txt`。

本次被 clone 的已提交 HEAD 为 b80337fd7aad8a37865f10e3f2cb6451d2775b96；
本轮修改按禁止提交的要求仍留在工作区。该回归确实测试 committed source 自身，没有模拟未提交 patch。
新工作区实现另行通过 19 文件 pre-cadence 严格审计；后续提交后同一测试会直接 clone 新 HEAD。
changed_files.csv 只记录 PRE_CADENCE_COMMIT→当前 HEAD 的已提交差异，不伪称包含本轮尚未提交修复。

## Cooldown 与固定规模回归

`python tools/validate_attack_cadence.py`：连续 25 个真实 simulation steps、deterministic miss，
实际 attempts **[1,11,21]**，3 次 attempt、22 blocked agent-steps，shape (4,52)。
首次合法 step 1 立即攻击，step 10 blocked、step 11 allowed。时钟仍为两侧 (4,) int64 next_fire_step。

完整 cadence 测试继续通过：exit/re-entry、target switch 不能绕过；hit/miss 均进入冷却；
reset 清零；到期无目标不延后 clock；每机独立；死亡机不攻击；Red/Blue 对称；
同 step mutual hit 同步结算；无限抽象弹药，无 actor fire action，无 cooldown observation。

新增固定规模测试确认 observation 固定 52/输出 (4,52)，scenario/factory 拒绝六机，
GroundAware 固定构造接口与 (4,) masks，MAPPO/MADSAC 52/3/4 pass、78/3/6 reject，
evaluator 六元素 return vector reject，正式 configs 无六机配置，legacy 不影响 import/pytest/runtime。

旧 v1 config/checkpoint 在 training、evaluation、MAPPO resume 和 checkpoint validation 中继续 fail-closed，
即使仍为 52D 也不能进入 v2。v2 config SHA 与 v1 不同，没有兼容 fallback 或自动迁移。

## 验证结果

完整正式测试：**420 passed in 40.46s**。
命令 `python -m pytest -q --basetemp=tmp/fixed_four_final`，日志 `reports/cadence_v2_fixed_four_pytest.txt`。
Cadence 专项验证、structural audit、独立 clean-clone regression 全部通过。
正式 env/algorithm/configs/README 共 **43 个文件，多波次关键词 0 命中**。
无可进入训练的 6v6 分支，无正式 FireState.armed 或 _entry_attempts 依赖。
Frozen weapon.py 中历史 FireState 类定义保留原样，但正式环境不导入/使用它。

两套 CUDA smoke 各 **1200 environment transitions**，使用 PowerShell uav Conda 环境：

| 方法 | 完成场次 | episode lengths | terminal reasons | 全部 reset seeds |
|---|---:|---|---|---|
| MAPPO | 9 | 147,118,125,117,117,115,124,115,111 | blue_win×6、red_win×3 | 0–9 |
| MADSAC | 10 | 125,106,118,124,116,96,106,115,118,147 | blue_win×9、red_win×1 | 5301–5311 |

两套 worker 均为 MultiUAVCombatEnv，Blue 为 GroundAwareNearestTargetPursuitPolicy，
52/3/4、max_steps=1000。每次终局后新 seed reset 与独立 reset 完全一致，双方 alive=4、steps=0；
保留的 terminal transition observation 与 reset observation 分离，seed 无复用，episode info 无 wave 字段。
Smoke 结果与上一轮相同，4v4 数值和 cadence 未变化。
证据：`outputs/fixed_four_v2_smoke/evidence.json`。

Checkpoint audit 验证 MAPPO 完整 v2 协议 evaluation，以及 MADSAC 严格 checkpoint protocol 和
deterministic/stochastic evaluation。结果保存在 `outputs/fixed_four_v2_checkpoint_audit/`。
Structural 19-file/config/archive 证据在 `outputs/fixed_four_v2_structural_audit/`。

**没有启动正式 3M 或其它长训练，没有执行提交、推送、分支或远程操作。**
只执行了本地 clean-clone 回归要求的复制式 clone，以及临时 clone 中明确要求的 frozen-file staging 测试。
最终正式 runtime 只有固定 4v4 single_round_v2；相对原正式 4v4 v1，唯一战斗语义变化仍为
exit-to-rearm → fixed 1.0 s per-aircraft cooldown。
