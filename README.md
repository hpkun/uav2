# Single-Round V2 UAV Combat

The only formal task is `single_round_v2`, environment version `2.3`: one 4v4
episode, 52D observation, 3D flight action, dt=0.1, max_steps=1000. Red or Blue
elimination terminates immediately; mutual elimination is a draw. Unresolved
combat truncates with `red_failure_timeout`. Vector auto-reset starts a new
independent 4v4 episode with a fresh seed and preserves the terminal transition.
The formal runtime is strictly fixed to four aircraft per side. Team-size
generalization is removed; six-aircraft scenarios, networks and return vectors
are rejected by the formal protocol and environment contracts.

Blue remains `ground_aware_nearest_target`. Dynamics, controller, RK4, aircraft
limits, radius=5000, 8 km opposing initialization, weapon geometry/hit model,
observation and R1–R4 reward parameters are unchanged by the cadence update.

## Attack cadence

The existing frozen weapon envelope determines eligibility, with the nearest
eligible target selected. First legal attack is immediate. Each attack attempt
starts a **1.0 s / 10-step cooldown**, independently for each aircraft on both
teams. Hit and miss both start cooldown. Switching targets or leaving/re-entering
the fire window cannot bypass it. An aircraft continuously inside the window
may fire again when its cooldown expires, without exiting the window. If there
is no eligible target at expiry, it waits and attacks as soon as one exists.
Ammunition remains abstract/unlimited; the actor does not choose fire actions.
Both sides compute attempts before simultaneous combat resolution.

An attempt at step 1 sets next_fire_step=11: steps 2–10 are blocked, step 11
is allowed. Continuous eligibility yields attempts at 1, 11, 21, ... . Reset
clears both integer per-aircraft clock arrays. Cooldown diagnostics appear in
info and never enter the 52D actor observation.

```yaml
attack:
  cadence: fixed_cooldown
  cooldown_seconds: 1.0
```

The underlying fire eligibility/hit model remains frozen; a 1.0 s per-aircraft
firing cooldown is introduced to avoid unrealistically repeated 0.1-s attack
attempts while removing the previous exit-to-rearm artifact.
**The 1.0 s cooldown is a project modeling extension**, not a specific cooldown
prescribed by the MADSAC paper.

## Training and validation

Use the uav Conda environment. CUDA is mandatory for training and checkpoint
audits/evaluation, with no CPU fallback. Formal MAPPO and MADSAC share the same
v2 validator and environment config. Low-level unit tests may use shorter time
limits. Networks and algorithm hyperparameters are unchanged; both use gamma=0.99.
Best checkpoint selection is win rate, average team return, then fewer Red losses.

```powershell
conda activate uav
python -m pytest -q
python tools/validate_attack_cadence.py
python tools/audit_single_round.py --output-dir outputs/cadence_audit
python tools/smoke_single_round.py --output-dir outputs/cadence_smoke
python tools/audit_single_round.py --output-dir outputs/cadence_audit --checkpoint-dir outputs/cadence_smoke
```

MAPPO defaults to configs/single_round_environment.yaml and configs/mappo.yaml.
MADSAC defaults to the same environment and configs/madsac_single_round_3m.yaml.

```powershell
python -u algorithm/train_mappo.py --env-config configs/single_round_environment.yaml --algorithm-config configs/mappo.yaml --device cuda --seed 0 --num-envs 24 --total-sampled-steps 3000000 --output-dir outputs/mappo_single_round_v2_seed0_3m
python -u algorithm/train_madsac.py --device cuda --output-dir outputs/madsac_single_round_v2_seed5301_3m
```

Historical v1 checkpoints/configs are rejected by formal factory, training,
evaluation and resume. There is no silent migration or compatibility bypass.
V1 and 6v6 materials are preserved under legacy/single_round_v1, outside formal
imports and default pytest collection. Existing historical archive files are
unchanged. Previous acceptance documents in reports describe their own versions.

The audit separates the historical archive baseline
8d09df1fd1e6d2425b712d0c336e9525cd313c4c from the immutable pre-cadence formal
baseline 89fd4c54d07205bd0952eb972cf133b4a41349f7. Nineteen frozen runtime files
must match pre-cadence Git blobs exactly, including observation, scenario,
GroundAware policy, common evaluator/vector environment, MAPPO runner, and both
networks/trainers. No team-size exceptions or self-declared baseline hashes are used.
The environment config may differ only in variant and the added attack section.
The original 435 archive files match the historical baseline; nine newer v1/6v6
snapshots match their committed archive identity b80337fd7aad8a37865f10e3f2cb6451d2775b96.
Existing legacy source content remains unchanged. changed_files.csv describes
pre-cadence→HEAD committed changes. Structural audit requires no local checkpoints
and writes ignored output files, leaving a clean checkout clean.

The clean-clone regression directly clones the local committed HEAD, audits it
twice, compares the JSON results and verifies cleanliness. It does not copy
uncommitted worktree files or create a temporary commit. Its deliberate frozen-file
staging check cannot change the immutable comparison baseline.

See reports/cadence_v2_acceptance.md for the detailed cadence acceptance results.
