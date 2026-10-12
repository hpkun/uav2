# Single-Round 架构收口验收报告

执行日期：2026-10-12。所有执行命令使用 PowerShell 激活
`D:\conda_envs\envs_dirs\uav`。训练、checkpoint 审计和 smoke 使用 CUDA，
设备为 NVIDIA GeForce RTX 5060 Laptop GPU，未回退 CPU。

## 正式协议

- 唯一正式 variant：`single_round_v1`；version：`2.3`。
- 每次 reset 生成 Red 4 架与 Blue 4 架，初始双方 alive=4；episode 内不补充飞机。
- Blue 全灭立即 Red win；Red 全灭立即 Blue win；双方全灭立即 draw，均 terminated。
- 未决达到 1000 步时 truncated，保持 `red_failure_timeout`。
- 终局后继续 step 明确报错，只能 reset 开始新 episode。
- Vector auto-reset 保留，新场使用新的 episode seed；真实终局 observation/mask
  保存在 transition 数据中，训练不把 reset observation 当作终局 observation。
- Observation/action/Red agent dimensions：52 / 3 / 4。
- Blue 正式使用 `GroundAwareNearestTargetPursuitPolicy`，配置类型为
  `ground_aware_nearest_target`；也明确支持 `nearest_target`，未知类型报错。
- GroundAware 六项 diagnostics 在 reset 清零，在 step info 持续提供。
- dt=0.1、radius=5000、8km 对置布局，以及 action/controller、aircraft、scenario、
  weapon、R1–R4、52D observation 参数均与成熟配置一致。

## 配置、训练和 checkpoint

MAPPO 默认：`configs/single_round_environment.yaml` + `configs/mappo.yaml`，gamma=0.99。
MADSAC 默认：同一环境 + `configs/madsac_single_round_3m.yaml`。
MADSAC 配置与旧配置内容完全一致：actor/critic lr=1e-4、gamma=0.99、alpha=0.1、
tau=0.001、replay=1,000,000、batch=1024、learning starts=10000、gradient steps=1、
policy delay=2、3M sampled environment transitions。

统一 best selection 为 `(win_rate, average_return, -average_red_loss)`。
旧 variant 或缺失 variant 均不能进入正式 factory；旧 checkpoint 不能正式 resume
或 evaluation。MAPPO 取消 cross-variant 参数及 CLI 开关，正式评估要求完整元数据、
匹配环境与算法配置指纹；MADSAC 保持 seed、身份、指纹及不支持 exact resume 的原协议。
两套方法共享单场任务指标汇总；MADSAC 保留 deterministic/stochastic evaluation。
正式训练 runner 与 checkpoint 评估入口明确拒绝 CPU。

## 修改文件与接口退出

正式修改/新增文件：

- `env/combat_env.py`、`env/factory.py`、`env/__init__.py`、`env/process_worker.py`
- `algorithm/common/vector_env.py`、`algorithm/common/evaluator.py`、`algorithm/common/checkpoint.py`
- `algorithm/mappo/runner.py`、`algorithm/mappo/evaluation.py`
- `algorithm/madsac/runner.py`、`algorithm/madsac/evaluation.py`、`algorithm/madsac/protocol.py`
- `algorithm/train_mappo.py`、`algorithm/train_madsac.py`、`algorithm/evaluate_mappo.py`、`algorithm/evaluate_madsac.py`
- `configs/single_round_environment.yaml`、`configs/madsac_single_round_3m.yaml`、`configs/mappo.yaml`
- 五个保留的基础测试文件及新增 `tests/test_single_round_protocol.py`
- `tools/smoke_single_round.py`、`tools/audit_single_round.py`
- `README.md`、`pyproject.toml`、`legacy/README.md` 和本报告及审计附件

完整逐文件清单见 [changed_files.csv](changed_files.csv)。完整归档映射与 Git blob
校验值见 [legacy_inventory.json](legacy_inventory.json)，共 435 个历史文件。

退出正式路径的接口：Persistent 环境 import/export；wave spawn/record/info；
curriculum export/restore worker commands 与 vector methods；`persistent_mission_metrics()`；
wave-based selection；per-wave 日志；Q2/Q3；Direct/Persistent transfer 正式入口。
worker 只保留 reset/step/close。未采用 total_waves=1 的兼容包装。

## 测试与 smoke

完整正式 pytest：**341 passed**。命令为
`python -m pytest -q --basetemp=tmp/single_round_pytest_report`，341 项通过，耗时 24.36 秒，结果见 `pytest.txt`。
历史研究测试已物理移至 legacy，默认测试路径只收集顶层 tests；没有重新接回 wave 逻辑。
基础 dynamics/controller/RK4、weapon、observation、reward 回归继续通过。
新增协议测试覆盖 factory 拒绝旧/缺失 variant、严格 4v4、三种全灭终局、禁止复活、
实际 1000 次未决 step 的 timeout、无 wave info、auto-reset 与 seed、GroundAware reset/override、
两套 evaluator 指标一致、checkpoint 拒绝旧 variant 和统一 selection 顺序。

两套 smoke 均使用正式环境参数、max_steps=1000，各采样 1200 transitions；
smoke 沿用现有缩小网络和训练规模的机制，MAPPO smoke rollout 设为 64，
评估分别为 1 / 2 场。正式算法配置未因 smoke 更改。

| 方法 | 已完成场次 | episode 长度 | terminal reason | 使用的 reset seeds |
|---|---:|---|---|---|
| MAPPO | 2 | 596, 471 | red_win, blue_win | 0, 1, 2 |
| MADSAC | 3 | 248, 583, 254 | blue_win, red_win, blue_win | 5301, 5302, 5303, 5304 |

每次 auto-reset 都逐项比较新 observation 与同一新 seed 的独立环境 reset，完全一致；
新 Red=4、Blue=4、steps=0，且与保留的 terminal observation 不同。
两套 smoke 均包含 Blue=0 的 Red win 终局，其后是新的完整 4v4 reset，非 Blue respawn。
MAPPO 执行 PPO 更新、评估和 checkpoint 保存；MADSAC 执行 critic/actor/target 更新、
独立 RNG 评估和 checkpoint 保存。smoke 是流程验收，不代表收敛或性能结论。

证据：`outputs/single_round_acceptance_final/evidence.json`，各方法子目录包含
training/optimization metrics、evaluation history、latest/best checkpoints 与配置快照。
追加 CUDA checkpoint 审计：MAPPO latest 的完整协议评估通过；MADSAC latest 的严格协议校验、
deterministic 和 stochastic 评估均通过。

## 关键词与稳定性审计

`env`、整个正式 `algorithm`、正式 `configs`、README 共 43 个文件，指定多波次关键词
命中 **0**。正式 Python 源码、测试与工具全部 AST 解析通过，`git diff --check` 通过。
测试中旧 variant/禁止字段的字符串只用于拒绝与缺失断言；审计工具中的关键词字面量只用于扫描。
其余研究语义位于 legacy 或历史 experiments；生成输出和临时目录不属于正式运行路径。

11 个底层 env 文件与 Git index 内容一致；两套网络与 trainer 源码保持一致；
核心环境参数与旧成熟配置逐节相等。435 个历史文件的归档内容按 Git 文本换行标准
校验一致。机器可读结果见 [single_round_audit.json](single_round_audit.json)。

## Legacy 文件与边界

- `legacy/env/persistent_env.py`：原 Persistent V1/V2 状态、spawn 与 curriculum 实现，供历史追溯。
- `legacy/algorithm/modular_mappo/`、`legacy/algorithm/modules/` 及两项 modular CLI：
  保留原多波次研究机制，不强行改写成单场版本。
- `legacy/configs/`：旧 Persistent、Direct、modular、wave-specific 与历史消融配置。
- `legacy/tests/`、`legacy/tools/`：旧研究机制与历史输出的测试、审计、可视化和实验工具。
- `legacy/README_original.md`：保留迁移前 README；`legacy/README.md` 为新归档说明。
- 原 `experiments/` 与论文材料保留用于历史实验追溯。

归档源文件保留原 imports 和路径用于追溯，不保证直接运行；复现旧研究应检出迁移前 Git 版本。
它们不被正式模块导入，不被默认 pytest 收集，也不属于新 README 的正式训练命令。
