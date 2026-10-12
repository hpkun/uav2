# single_round_v1 最后一轮协议收口验收

本轮只修改公共正式协议、MAPPO checkpoint 校验、审计复现机制及对应测试与运行工具。
使用 PowerShell 的 `D:\conda_envs\envs_dirs\uav` 环境；CUDA 为 NVIDIA GeForce RTX 5060 Laptop GPU。

## 实际修改

正式代码文件：

- `algorithm/common/protocol.py`：新增公共 `validate_single_round_environment_config()`。
- `algorithm/mappo/runner.py`：初始化正式协议校验，resume 使用实际运行 seed/env count/smoke 身份。
- `algorithm/madsac/protocol.py`：复用公共环境校验，保留算法维度、seed、checkpoint 等原有严格检查。
- `algorithm/common/evaluator.py`、`algorithm/madsac/evaluation.py`：直接调用评估器也必须通过正式环境校验。
- `algorithm/common/checkpoint.py`：resume 必需字段缺失或不匹配均拒绝，evaluation 指纹也 fail-closed。
- `algorithm/train_mappo.py`：先解析原运行身份再严格验证；第二层不再跳过缺失字段。

工具：`tools/audit_single_round.py` 改为显式 baseline 审计，结构审计无需本地 checkpoint；
`tools/smoke_single_round.py` 新增输出目录参数，允许在新目录复跑。
README 补充正式协议和可复现命令。未修改任何 env 源码、正式配置、网络、trainer 或训练超参数。

测试文件：

- 新增 `tests/test_formal_single_round_closure.py`，43 个测试实例。
- 修改 `tests/test_mappo.py`，正式 runner 测试不再缩短环境时限。
- 修改 `tests/test_training_protocol.py`，正式 evaluator 使用 1000 步配置；底层 vector 的短时限测试保留。
- 修改 `tests/test_single_round_protocol.py`，指标一致性测试改为完整合法正式 episode。
- 修改 `tests/test_madsac.py`，RNG 隔离测试传入正式环境配置，mock 的 episode 消耗逻辑保留。

## 公共 validator 与身份校验

公共 validator 固定以下协议：

| 字段 | 要求 |
|---|---|
| environment_version | 字符串 `2.3` |
| environment_variant | `single_round_v1` |
| simulation.dt | 0.1 |
| simulation.max_steps | 1000 |
| scenario.team_size | 4 |
| observation_dim | 通过现有 `observation_dim_from_config()` 计算，必须为 52 |
| action_dim | 正式环境固定为 3；显式配置声明不一致时拒绝 |

缺失 simulation/scenario/observation/action 配置段也拒绝。MAPPO/MADSAC 算法配置中的
obs/action/agent 维度继续分别验证为 52/3/4。Smoke 不绕过环境协议。
底层 `MultiUAVCombatEnv` 未写死时限，仍允许测试用 max_steps=1；短时限底层 timeout 回归通过。

MAPPO formal resume 必需且匹配的元数据：environment_version、environment_variant、
environment_config_sha256、algorithm_config_sha256、training_seed、training_num_envs、
training_gamma、training_smoke。实现版本与已有维度/critic 检查继续保留。
Seed/env count/smoke 与原运行身份或明确的 runner runtime 比较，而非以 checkpoint 自己的值自证。
CLI 中原 run_config 的第二层校验也要求每个字段存在且相等；配置快照不能替代 checkpoint 元数据。
没有增加 legacy resume、force 或忽略不匹配的旁路。

## baseline 审计

固定 baseline：`8d09df1fd1e6d2425b712d0c336e9525cd313c4c`。
使用 `git ls-tree -r BASE_COMMIT` 获取历史 blob，用 `git show BASE_COMMIT:path`
读取历史配置；与当前工作树文件比较，按 Git 文本 CRLF/LF 规则计算 blob hash。
不再使用 `git ls-files -s` 作为旧基准，也不依赖 dirty 工作区获取变更清单。

`changed_files.csv` 使用 `git diff --name-status -M BASE_COMMIT HEAD`，描述已经提交的差异。
本次主仓改动尚未提交，因此当前清单的 HEAD 为 `7bba545f41809829ee67009aac6779685638e58e`，
不冒充包含本轮未提交改动；提交后同一命令会自然生成新的 BASE→HEAD 清单。
本轮修改文件已在本报告逐项列出。

Legacy 每个文件映射回 baseline 原路径并验证内容；只有新 `legacy/README.md` 归档索引豁免，
`README_original.md` 映射到原 README。缺失归档或无 baseline 对应的非豁免文件报错。
**435 个 legacy 文件全部通过**，未重写历史研究代码。

结构审计默认写到被 Git 忽略的 `outputs/single_round_audit`，不需要 smoke checkpoint，
所以 clean clone 运行后仍保持 clean。`--checkpoint-dir` 是附加 CUDA 审计，不改变结构验收规则。
审计断言采用显式异常，Python 优化模式也不会跳过这些校验。

自动测试在临时 clone 内提交本轮代码快照，确认提交后状态 clean；连续两次结构审计输出完全相同，
运行后仍 clean；再修改并暂存 dynamics 文件，冻结文件比较仍拒绝，证明 index 不能重定义 baseline。
主仓未被该测试提交或暂存。

## 最终结果

正式 pytest：**384 passed in 36.67s**。命令：
`python -m pytest -q --basetemp=tmp/protocol_close_pass`；日志为 `reports/protocol_closure_pytest.txt`。

覆盖非法正式环境、smoke 环境身份、MADSAC 严格维度、八项 checkpoint 元数据逐项删除/篡改、
合法 checkpoint、evaluation 缺失指纹、CLI 配置快照不能兜底，以及真实最后一架 Blue 被击杀的 step。
最后击杀测试只固定现有 hit-resolution hook，未修改武器参数；本 step terminated=True、
truncated=False、Blue=0、red_win，返回观测的 Blue 槽为空，未生成新 Blue，再 step 明确报错。

关键词审计范围：env、整个 algorithm、正式 configs、README，共 **43 个文件，0 命中**。
测试里的拒绝字面量、工具扫描词表及 legacy 不计入正式 runtime 指标。
正式源码、测试及工具 AST 检查通过，`git diff --check` 通过。

## 复跑 CUDA smoke

命令：`python tools/smoke_single_round.py --output-dir outputs/protocol_closure_smoke`。
两套均采样 1200 个 environment transitions，保留正式 1000-step 环境与原 smoke 成本缩减机制。

| 方法 | sampled_steps | 完成场次 | episode lengths | terminal reasons | 全部 reset seeds |
|---|---:|---:|---|---|---|
| MAPPO | 1200 | 2 | 596, 471 | red_win, blue_win | 0, 1, 2 |
| MADSAC | 1200 | 3 | 248, 583, 254 | blue_win, red_win, blue_win | 5301, 5302, 5303, 5304 |

两套 worker environment 均为 `MultiUAVCombatEnv`，Blue policy 均为
`GroundAwareNearestTargetPursuitPolicy`。每个终局后的 observation 与新 seed 独立 reset 完全一致，
新 Red=4、Blue=4、steps=0；保留的 terminal transition observation 与 reset observation 不同。
Reset seed 无复用；completed episode info 无 wave 字段。两套均包含 Blue 全灭的 Red win 终局，
随后是新的 4v4 episode，无中途 Blue respawn。

追加审计命令：
`python tools/audit_single_round.py --output-dir outputs/protocol_closure_audit --checkpoint-dir outputs/protocol_closure_smoke`。
MAPPO 完整协议 checkpoint evaluation、MADSAC 严格 checkpoint validation 及两种 evaluation 模式均通过。
证据位于 `outputs/protocol_closure_smoke/evidence.json` 及
`outputs/protocol_closure_audit/single_round_audit.json`，同目录包含 BASE→HEAD 文件清单和 legacy inventory。

## 冻结内容与实际边界

相对指定 baseline，11 个 env 基础文件均保持内容一致：control、dynamics、integrator、weapon、
reward、observation、scenario、fixed_policy、models、geometry、math_utils。
尤其 `fixed_policy.py` 的整个内容一致，GroundAware 核心行为未变。
MAPPO/MADSAC 的 networks.py 与 trainer.py 共 4 个文件也完全一致。
正式配置中被冻结的 action、aircraft、arena、scenario、weapon、reward、observation、dt、
GroundAware 参数及 desired_speed 均与 baseline 成熟配置一致；MADSAC 正式配置完全一致。

没有发现需要回滚的冻结文件冲突。实际需区分的是：结构审计不能依赖 clone 中不存在的忽略 checkpoint，
所以将其与可选 CUDA checkpoint 审计分开；HEAD 变更清单按要求只记录已提交内容。
这些是最小工具边界调整，未更改环境或算法研究定义。
