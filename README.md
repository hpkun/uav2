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
time-to-ground <=4 s, forces beta=0 and L=5, and keeps thrust legal. This matches
the previous guard's 2*2 s time scale. A one-ULP tolerance handles threshold rounding.

`configs/pursuit_environment.yaml` defines the independent `pursuit_v1` task.
`algorithm/train_pursuit.py` uses the unchanged MAPPO network/PPO trainer.
It shares direct controls, integration and the 52D observation builder, has
random legal initial positions and seeded random opponent controls held for
20 steps, and has no weapon or attack configuration/object/path.
Reward is exclusively `-nearest_distance_m * distance_reward_scale`, default 1.0.
Dead-agent rewards are zero under the existing alive-mask architecture; raw
distance validity masks are logged explicitly. The 1000-step horizon is a
**project implementation choice**, not a reported paper horizon.

Pursuit logs raw per-agent distance/reward, mean/final/minimum nearest distance,
ground/boundary losses, mean speed, speed/pitch envelope hit ratios, mean absolute
T/L/beta and near-limit fractions. Near-limit means >=95% of the physical limit.
Distance means/minima use valid alive-agent samples; final distance uses each
agent's last valid distance, avoiding a zero-distance artifact after elimination.
State/control diagnostics sample Red UAVs alive before each integration step.
These diagnostics never enter reward. Pursuit checkpoint selection maximizes
mean team return; combat keeps win rate, return, then fewer Red losses.

MAPPO supports `--init-actor-checkpoint` for **fresh combat v3 runs only**.
It requires a pursuit checkpoint with version/task/action identity, 52/3/4
dimensions and exactly compatible actor tensor keys/shapes. It loads only actor
parameters and records source path/SHA-256/task/action semantics. Critic,
optimizers, counters and history stay fresh. It cannot be combined with resume.
Normal resume/evaluation require exact task/config hashes and direct-control
identity. Smoke actors use hidden=64; formal actors use the unchanged hidden=256,
so smoke and formal actors cannot initialize one another.

Runtime policy: use WSL Ubuntu with `conda activate uav`; if WSL is unavailable,
use Windows PowerShell's uav environment. CUDA is mandatory for training and
checkpoint audits/evaluation; there is no CPU fallback. Repository tests and
static checks also run in uav. Validation tools are `tools/audit_single_round.py`,
`tools/validate_attack_cadence.py`, and `tools/smoke_direct_control.py`.
The latter caps each of combat, pursuit and actor-initialized combat at 1200
environment transitions and refuses nonempty output directories.

The audit compares 16 frozen sources against immutable v2 Git blobs, checks
frozen combat method ASTs and preserved legacy controller definitions, scans
the active Red execution/decoder path for old controller dependencies, and
verifies the historical archives. Its changed-files CSV remains explicitly a
committed pre-cadence-to-HEAD inventory; working-tree changes are audited separately.
The clean-clone regression tests committed HEAD without copying local patches
or creating a temporary commit. Long training awaits user confirmation.

See `reports/direct_control_v3_acceptance.md` for this iteration's results.
