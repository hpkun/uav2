# single_round_v2 攻击节奏验收

本轮按用户明确答复，将正式任务从上轮 6v6/78D 恢复为 **4v4/52D**。
v1 和 6v6 材料新增归档到 `legacy/single_round_v1/`；原有 legacy 文件未改。
在原 4v4 配置基础上，只有 variant 与 attack 配置变化。底层已有团队大小泛化源码
保留原样，没有为了本轮恢复 4v4 去改写 observation/scenario/Blue policy。

## 文件与实现

本轮正式修改：`env/combat_env.py`、`env/config.py`、`env/factory.py`、`env/__init__.py`、
`algorithm/common/protocol.py`、`algorithm/madsac/protocol.py`、
`algorithm/train_mappo.py`、`algorithm/evaluate_mappo.py`、`configs/single_round_environment.yaml`。
记录工具 `tools/record_single_round_match.py` 的观察 hook 改为新方法名；
`tools/audit_single_round.py` 增加本轮冻结和新归档校验；新增 `tools/validate_attack_cadence.py`。
README 改为正式 v2 协议。没有修改网络/trainer 或算法超参数配置。

新增 `tests/test_attack_cadence.py`；旧 `test_combat_environment.py` 的 entry/rearm 测试
重写为连续窗口 cooldown 回归，未删除。`test_single_round_protocol.py` 更新正式 variant
并增加旧 v1/6v6 拒绝用例。clean-clone 审计测试补充复制新冻结清单与归档。
6v6 专用测试保留于新历史归档，不作为正式 4v4 测试执行。

## 精确 cadence

使用 `red_next_fire_step`、`blue_next_fire_step`，shape 均 `(4,)`、dtype `int64`。
reset 全部归零。运动完成后 steps 增加 1，在攻击结算时检查整数 clock。
存活攻击机存在合法目标且 `steps >= next_fire_step[i]` 时，对最近合法目标进行一次 attempt；
无论 hit 或 miss，写入 `next_fire_step[i] = steps + 10`。

首次合法 step=1 立即攻击，随后 next_fire_step=11；step 2–10 不允许，step 11 允许，
没有延迟到 step 12。持续合法无需离窗再进入。
无合法目标时 clock 不被清除或延后；死亡机不攻击；无弹药约束，无 fire actor action。
双方 attempt 先分别生成，再沿用原 combat resolver 同步结算。

正式配置新增独立顶层 `attack.cadence=fixed_cooldown`、`attack.cooldown_seconds=1.0`。
WeaponEnvelope 接收的 weapon 配置原样保留。公共 validator 要求 v2、2.3、dt=0.1、
1000 steps、4 agents、52D/3D，并验证整数换算得到 cooldown_steps=10。
非正冷却、非整数换算、错误 cadence、非 1 秒冷却均拒绝。
底层短 max_steps 测试仍可使用合法 v2 配置。

保留全部现有 firing/combat metrics，新增两侧 step_cooldown_blocked、
cooldown_blocked_agent_steps 累计值及 next_fire_steps 副本。
只在存活机当前有合法目标但 clock 未到期时计 blocked agent-step。
这些 diagnostics 不加入 observation。

## 专项结果

`python tools/validate_attack_cadence.py` 在未冻结运动的 25 个真实 simulation steps 中，
仅固定现有 hit hook 为 miss，实际输出 **[1, 11, 21]**，3 次 attempt，22 个 blocked agent-steps，
observation shape `(4,52)`。证据：`outputs/cadence_v2_validation/cadence.json`。

专项自动测试确认：

- 首次立即攻击；step 10 blocked、step 11 allowed，持续窗口可以再次攻击。
- step 2–5 离窗、step 6 回到窗口不会提前 rearm，到 step 11 才能攻击。
- 更换 Blue 目标不能绕过同一 Red 攻击机 clock。
- hit/miss 均写入 step+10；目标死亡也保留已写入 clock。
- reset 清零；冷却到期没有目标不延后 clock，目标重现后立即攻击。
- 每机 clock 独立；死亡机不攻击；nearest eligible 目标选择未变。
- Red/Blue 同等几何的 attempt 序列均为 [1,11,21]。
- deterministic mutual hit 时双方均生成 attempt，双方均被击杀并 draw，没有先手消除反击。
- cooldown 数组变化不改变 observation；原 R3/R4 源码数值比较一致；R1/R2 回归通过。

## 冻结与协议审计

**Observation 仍为 52D，未增加 ready/cooldown 特征。R1–R4 数值、公式与结算位置未改。**
本轮全部 11 个环境基础文件加四个 MAPPO/MADSAC networks/trainer 文件，与
`reports/cadence_v2_baseline.json` 记录的修改前 Git 文本 blob 一致。
其中 `weapon.py` 未编辑，geometry、hit_threshold、Gaussian sampling、射程与 true-3D gate 均未变。
control、dynamics、integrator、aircraft、arena、scenario、Blue policy 同样未因本轮变化。

历史 baseline 仍为 `8d09df1fd1e6d2425b712d0c336e9525cd313c4c`。
Observation/scenario/fixed_policy 与历史 initial baseline 的差异来自上一轮获授权的团队大小泛化，
审计明确显示这些既有差异；本轮 15 文件冻结校验另行严格证明没有追加修改。
不存在把这三个文件伪称为与 initial baseline byte-identical 的情况。
新归档 9 个文件均与修改前快照 hash 匹配，原 435 个历史文件仍匹配原 baseline。

正式 runtime 仅允许 `single_round_v2`，factory 拒绝 v1 和 6v6 v1；
MAPPO/MADSAC 的 training、evaluation、checkpoint validator 共用 v2 定义。
旧 v1 checkpoint 即使同为 52D，也因 variant/config SHA 不符被明确拒绝。
没有 silent compatibility、自动迁移或 legacy resume 旁路。
正式 combat_env 不 import/use FireState，不使用 `.armed`、red/blue_fire_states 或 `_entry_attempts`。
冻结 weapon.py 保留历史 FireState 类定义，但不被正式环境导入。

正式 env/algorithm/configs/README 共 **43 个文件，多波次关键词 0 命中**。
源码、测试、工具 AST 检查通过；clean-clone 连续结构审计可复现；git diff --check 通过。
审计证据：`outputs/cadence_v2_audit/single_round_audit.json`。

## 正式测试与 CUDA smoke

完整正式 pytest：**411 passed in 58.08s**。
命令 `python -m pytest -q --basetemp=tmp/cadence_v2_final`，日志 `reports/cadence_v2_pytest.txt`。
最后补强缺失 attack 配置段与 bool 冷却值的 fail-closed 后，cadence/protocol 专项复查
**66 passed in 13.26s**；最终结构审计再次通过。

两套 CUDA smoke 均采样 **1200 environment transitions**：

| 方法 | 完成场次 | episode lengths | terminal reasons | 全部 reset seeds |
|---|---:|---|---|---|
| MAPPO | 9 | 147,118,125,117,117,115,124,115,111 | blue_win×6、red_win×3 | 0–9 |
| MADSAC | 10 | 125,106,118,124,116,96,106,115,118,147 | blue_win×9、red_win×1 | 5301–5311 |

两套 worker class 均为 MultiUAVCombatEnv，Blue policy 均为 GroundAwareNearestTargetPursuitPolicy；
4v4、52D/3D、max_steps=1000，episode info 无 wave 字段。
每次终局之后，新 seed 的完整 reset observation 与独立 reference reset 完全一致，
双方恢复 alive=4、steps=0；terminal transition observation 与 reset observation 不混淆。
Seed 无复用；auto-reset 与 terminal transition 语义未修改。

证据：`outputs/cadence_v2_smoke/evidence.json`。
追加 CUDA 审计也通过 MAPPO 完整协议 checkpoint evaluation、MADSAC 严格 checkpoint 检查与
deterministic/stochastic evaluation。Smoke 沿用现有降低网络/rollout/evaluation 成本的机制，
正式算法配置未调参。

**没有启动任何正式长训练。** 本轮仅运行两个 1200-step smoke、测试、专项 cadence 验证和审计。
1 秒冷却是项目建模扩展，没有声称该数值由 MADSAC 论文规定。
