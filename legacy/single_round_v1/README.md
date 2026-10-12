# Single-Round 6v6 UAV Combat

MAPPO now defaults to `single_round_6v6_v1`, environment version `2.3`. Each reset
creates six Red and six Blue aircraft. No aircraft are replenished during
an episode. Elimination of either team terminates the episode; simultaneous
elimination is a draw. Unresolved combat truncates at 1000 steps with
`red_failure_timeout`. Vector auto-reset starts a new independent 6v6 episode
using a fresh seed and preserves the terminal transition for learning.

Blue uses `ground_aware_nearest_target`, with episode-level guard diagnostics.
The 78D observation includes self + five allies + six enemies. The 3D action,
dt=0.1, radius=5000, initial
8 km opposing formation, controller, dynamics, RK4, weapon and R1–R4 reward
parameters are retained. Both baselines use gamma=0.99 and select their best
checkpoint by win rate, team return, then fewer Red losses.

Run commands in PowerShell after activating the `uav` Conda environment.
CUDA is required for training and checkpoint evaluation; CPU fallback is rejected.

```powershell
conda activate uav
python -m pytest -q
python algorithm/train_mappo.py --device cuda --output-dir outputs/mappo_single_round
python algorithm/train_madsac.py --device cuda --output-dir outputs/madsac_single_round
```

MAPPO defaults to `configs/single_round_6v6_environment.yaml` and `configs/mappo_6v6_3m.yaml`.
Its 3M environment-transition command (PowerShell) is:

```powershell
python -u algorithm/train_mappo.py --env-config configs/single_round_6v6_environment.yaml --algorithm-config configs/mappo_6v6_3m.yaml --device cuda --seed 0 --num-envs 24 --total-sampled-steps 3000000 --output-dir outputs/mappo_single_round_6v6_seed0_3m
```

The original 4v4 `single_round_v1` configuration is retained for existing models:
`configs/single_round_environment.yaml` with `configs/mappo.yaml`.
The 4v4 checkpoints have 52D inputs and cannot resume into the 78D 6v6 task.
MADSAC defaults remain the 4v4 environment with `configs/madsac_single_round_3m.yaml`.
MADSAC retains its existing hyperparameters and 3M environment-transition target.

```powershell
python algorithm/train_mappo.py --smoke --device cuda --num-envs 1 --total-sampled-steps 16 --output-dir outputs/mappo_smoke
python algorithm/train_madsac.py --smoke --device cuda --num-envs 1 --total-sampled-steps 16 --output-dir outputs/madsac_smoke
python algorithm/evaluate_mappo.py --checkpoint outputs/mappo_single_round/best_eval.pt --device cuda --seed-base 10000000 --episodes 20 --output outputs/mappo_eval.json
python algorithm/evaluate_madsac.py --checkpoint outputs/madsac_single_round/best_eval.pt --device cuda --mode stochastic --output outputs/madsac_eval.json
python algorithm/evaluate_madsac.py --checkpoint outputs/madsac_single_round/best_eval.pt --device cuda --mode deterministic --output outputs/madsac_eval_deterministic.json
```

Formal evaluation requires matching checkpoint protocol and configuration
fingerprints. Historical checkpoints cannot resume or enter formal evaluation.
MADSAC retains its existing actor-only evaluation and unsupported exact resume
contract; MAPPO retains continuation with fresh episode seeds.

Historical research sources and experiments are documented in
[legacy/README.md](legacy/README.md), outside formal imports and test collection.
The acceptance report and exact file inventory are in `reports/`.

Formal training and evaluation share one validator for version 2.3, dt=0.1,
max_steps=1000, and exact task dimensions: 78/3/6 for single_round_6v6_v1,
52/3/4 for the original single_round_v1. Smoke runs keep their environment contract. Low-level environment unit
tests may still use shorter time limits. MAPPO resume requires complete matching
configuration fingerprints, seed, environment count, gamma, and smoke identity.

The reproducible audit uses baseline commit
`8d09df1fd1e6d2425b712d0c336e9525cd313c4c`. Structural audit works after a clean
clone without local smoke checkpoints and writes ignored output files:

```powershell
python tools/audit_single_round.py
python tools/smoke_single_round.py --output-dir outputs/protocol_smoke
python tools/audit_single_round.py --checkpoint-dir outputs/protocol_smoke
```

`changed_files.csv` describes committed baseline-to-HEAD changes. Frozen source
and archived-content checks compare baseline blobs with the current files and
ignore the Git index. Optional checkpoint audit requires CUDA. See
`reports/protocol_closure_acceptance.md` for the final protocol acceptance results.
That report describes the preceding 4v4 closure. Current audit output explicitly
reports the user-authorized team-size refactors in observation/scenario/fixed-policy
sources; dynamics, controller, RK4, weapons, rewards, networks and trainers remain frozen.
