# Direct Physical Control UAV Tasks

The formal combat environment is `single_round_v3`, version `2.4`, with
`task_type=combat` and `action_semantics=direct_physical_control`. It retains
4v4, 52D observations, dt=0.1 s, and a 1000-step horizon. Historical v1/v2
configs and checkpoints are rejected by formal training, evaluation and resume.
The frozen v2 source reference is commit `81d623e1ecafbf7df91aa2d9dc1fc19cf885f2a9`;
selected source/config snapshots are under `legacy/single_round_v2`.

The actor produces normalized `[a_T, a_L, a_beta]`. After clipping to [-1,1],
the decoder returns `ControlCommand(nx=2*a_T, nz=5*a_L, phi=(pi/3)*a_beta)`.
It performs no tracking, inverse dynamics, load projection, smoothing or bias.
Zero action gives zero lift and therefore dtheta/dt=-g/v at theta=0.
Horizontal trim is physical `[0,1,0]`, normalized `[0,0.2,0]`.
Old delta controller APIs remain available for historical analysis.

PointMassDynamics, RK4, speed/pitch envelopes, combat scenario/initialization,
observation features, weapon geometry/hit model and R1–R4 are frozen. The
4000 m weapon range and independent 1 s / 10-step firing cooldown remain:
continuous eligibility produces attempts at steps 1,11,21,... . Hit and miss
both start cooldown; both sides attempt before simultaneous resolution.

Blue uses deterministic nearest-alive-target guidance and desired speed 250 m/s.
Response times under `blue_policy` describe speed acceleration and heading/pitch
error responses (seconds); rate bounds are rad/s. Only Blue inverts requested
angular rates to physical lift/roll, clips to physical limits, and emits the same
normalized physical action decoded for Red. Ground protection explicitly uses
time-to-ground <=4 s using the larger of current and commanded downward speeds,
forces beta=0 and L=5, and keeps thrust legal. This matches
the previous guard's 2*2 s time scale. A one-ULP tolerance handles threshold rounding.
`consider_commanded_pitch: true` is mandatory. Combat remains v3, but its config
fingerprint changed before formal training; old v3 smoke checkpoints are rejected.

`configs/pursuit_environment.yaml` defines the independent `pursuit_v2` task.
`algorithm/train_pursuit.py` uses the unchanged MAPPO network/PPO trainer.
It shares direct controls, integration and the 52D observation builder, has
random legal initial positions and independent seeded random waypoints sampled
every 50 steps. Opponents share Blue's direct guidance toward these waypoints;
they do not teleport. Waypoints are area-uniform in a 2500 m disk, altitude
2200..3800 m, with target speed 200..250 m/s. Opponent-only boundary guidance
aims at the arena center at radius >=3000 m, altitude 3000 m and speed 200 m/s.
The task has no weapon or attack configuration/object/path.
Reward is exclusively `-nearest_distance_m * 0.001`, the negative distance in km.
The validator requires exactly `distance_reward_scale: 0.001`. All eight UAVs
remain alive, all masks stay one, and all Red receive real distance rewards
on every step. Ground/boundary crossings are diagnostics only. Terminated is
always false; truncated is true exactly at step 1000. The 1000-step horizon is a
**project implementation choice**, not a reported paper horizon.

Pursuit logs raw per-agent distance/reward, mean/final/minimum nearest distance,
ground/boundary violation agent-steps, distinct violating agents and rates for
Red/opponents, mean speed, speed/pitch envelope hit ratios, mean absolute
T/L/beta and near-limit fractions. Near-limit means >=95% of the physical limit.
Distance means/minima use all four agents at every step; final distance is the
four-agent mean at the final step. Violation rates divide agent-steps by 4*steps.
State/control diagnostics sample Red UAVs alive before each integration step.
These diagnostics never enter reward. Pursuit checkpoint selection minimizes
mean nearest distance, then final nearest distance; return and minimum distance
do not select best. Combat keeps win rate, return, then fewer Red losses.

MAPPO supports `--init-actor-checkpoint` for **fresh combat v3 runs only**.
It requires a pursuit_v2 checkpoint with version/task/action identity, 52/3/4
dimensions and exactly compatible actor tensor keys/shapes. It loads only actor
parameters and records source path/SHA-256/task/action semantics. Critic,
optimizers, counters and history stay fresh. It cannot be combined with resume.
Normal resume/evaluation require exact task/config hashes and direct-control
identity. Flawed pursuit_v1 checkpoints are rejected for resume, evaluation and
actor initialization; snapshots are retained under legacy/pursuit_v1.
Smoke actors use hidden=64; formal actors use the unchanged hidden=256,
so smoke and formal actors cannot initialize one another.

Runtime policy: use WSL Ubuntu with `conda activate uav`; if WSL is unavailable,
use Windows PowerShell's uav environment. CUDA is mandatory for training and
checkpoint audits/evaluation; there is no CPU fallback. Repository tests and
static checks also run in uav. Validation tools are `tools/audit_single_round.py`,
`tools/validate_attack_cadence.py`, and `tools/smoke_direct_control.py`.
The latter caps each of combat, pursuit and actor-initialized combat at 1200
environment transitions and refuses nonempty output directories.
`tools/validate_pursuit_waypoints.py` runs 100 seeds with 1000 steps each, using
Red trim actions, and requires zero opponent ground/boundary violations.

The audit compares 16 frozen sources against immutable v2 Git blobs, checks
frozen combat method ASTs and preserved legacy controller definitions, scans
the active Red execution/decoder path for old controller dependencies, and
verifies the historical archives. Its changed-files CSV remains explicitly a
committed pre-cadence-to-HEAD inventory; working-tree changes are audited separately.
The clean-clone regression tests committed HEAD without copying local patches
or creating a temporary commit. Long training awaits user confirmation.

See `reports/direct_control_v3_acceptance.md` for this iteration's results.

## Training metrics persistence

MAPPO combat/pursuit and MADSAC write training_metrics.jsonl snapshots at their
existing runtime_logging.console_interval_sampled_steps (currently 20000), with
an independent threshold counter. Threshold crossing handles non-divisible
vector sizes; final actual sampled_steps is always recorded without duplicates.
Snapshots preserve the original per-transition schema, including pursuit arrays;
they are not interval averages. JSONL uses compact separators, without gzip.
Optimization metrics, evaluation history, console logging and best selection
keep their previous cadence and content. Configs and checkpoint identities do
not change. MAPPO resume accepts existing high-frequency history and reloads
the last legal log step after rollback. MADSAC exact resume remains unsupported
under its existing replay/checkpoint protocol.

`tools/compact_training_metrics.py` is an optional streaming tool for completed
old logs. Use `--input`, optional `--output`, and optional
`--interval-sampled-steps` (default 20000). It retains first, threshold-crossing
and final records. Default output is a new training_metrics.compact.jsonl file;
existing output paths and source overwrite are rejected. Only explicit
`--replace` creates a synced temporary file, backs up the original, and atomically
installs the compact file. Replacement is refused when source size/mtime/identity
changes during scanning. Invalid sampled_steps or non-increasing inputs are
rejected. No existing experimental logs are automatically compacted.

`tools/validate_training_metrics_logging.py` runs bounded CUDA smokes and the
3M-equivalent synthetic counter test in a fresh output directory. See
`reports/training_metrics_logging_acceptance.md` for logging-only acceptance.
