# V1 — Camera-equipped FR3 reaching and IK expert

V1 implements the first simulation backend for the V0
`fr3_visual_reach_v1` policy I/O contract. It does not run SmolVLA, collect a
training dataset, train PPO, perform domain randomization or connect hardware.
LeRobot/Transformers remain in the isolated learning environment.

## Task and physical contract

| Item | Definition |
|---|---|
| Gym task | `Lab-VisualReach-FR3-v0` |
| Robot | Audited FR3v2 bare flange, fixed base, no gripper |
| State | Seven measured joint positions in radians, `panda_joint1..7` |
| Cameras | Overhead and wrist RGB, 256×256, uint8 HWC; batch dimension in simulator |
| Instruction | “Move the end effector to the red marker.” |
| Action | `[dx, dy, dz]` **meters**, robot-base frame, L2 norm ≤ 0.005 m |
| Reference | Current measured flange position **at each action execution**, not accumulated references |
| Orientation | Episode-initial flange orientation held by full-pose DLS |
| Rates | Physics 200 Hz; action/camera observations 20 Hz (decimation 10) |
| Goal volume | x ∈ [0.35, 0.46], y ∈ [−0.08, 0.08], z ∈ [0.40, 0.53] m, base frame |
| Reset | Exact home joint position, zero joint velocity, fresh independent uniform target |
| Success | Error < 0.02 m for five consecutive policy steps |
| Timeout | 10 s; success and invalid state are terminations, timeout is truncation |

Policy observations contain **only q and the two RGB images**; the bridge adds
language. Target coordinates are available to the scripted expert, reward,
termination and diagnostics, never to policy features. The marker is an actual
red kinematic, collision-disabled sphere, visible to RTX cameras—not a GUI-only
debug marker. Self-collision physics is enabled; there is no collision reward
or separately qualified collision-safety controller.

The expert commands

$$
\Delta p=0.3(p_g-p),\qquad
\Delta p_{\mathrm{bounded}}=\Delta p\min\left(1,\frac{0.005}{\|\Delta p\|}\right).
$$

The zero-error case returns zero. The action term rejects nonfinite/oversized
inputs instead of silently rescaling them. Full-pose DLS computes joint position
targets every physics step; those targets are projected to the audited command
bounds. Projection is reported separately. See [asset/controller provenance](../source/isaac_tasks/isaac_tasks/assets/fr3/README.md).

The wrist sensor is a virtual outboard mount with a wide-angle lens, not a
calibrated physical camera. V2/dataset collection must qualify visibility over
the intended distribution and replace this geometry if matching hardware.
The pinned PhysX/Fabric checkout did not propagate articulated link motion to
the child camera's world transform reliably. Its prim is therefore outside the
articulation hierarchy. The first policy observation term sets its world pose
from the measured flange pose and fixed mount **before either camera renders**,
then advances the render generation. Verification checks both position and rotation mount parity
throughout the rollout; a changing frame counter alone does not prove mounting.

## Implementation map

| File (under `source/isaac_tasks/isaac_tasks/`) | Responsibility |
|---|---|
| `assets/fr3/model.json`, `assets/fr3/visual_reach.py` | Audited model metadata, explicit asset resolution and checksum, articulation/PD config |
| `tasks/manager_based/reach/config/fr3/visual_reach_env_cfg.py` | Scene, cameras and all manager configuration groups |
| `tasks/manager_based/reach/config/fr3/__init__.py` | Complete environment registration; lazy config loading |
| `tasks/manager_based/reach/mdp/visual_reach.py` | Visible target sampling, RGB/q observations, reset, reward and success/failure terms |
| `controllers/visual_reach.py` | Measured-TCP increment action, batched DLS and privileged expert |
| `controllers/visual_reach_checks.py` | Finite-difference link FK/Jacobian and known-action clone trace |
| `bridges/visual_reach.py` | Typed-contract validation, device-resident feature mapping, camera counters/extrinsics/intrinsics |

`scripts/run_visual_reach_expert.py` performs bounded verification and reports
episode outcomes. `tests/test_visual_reach.py` tests offline semantics. Camera
timestamps, IPC, episode-aware action chunks and recording/export are V2 work.
Do not feed the pretrained model's native 6D action output to this 3D interface.

## Run on this workstation

The two local editable packages are installed without changing Isaac dependencies.
The inspected `isaac-tasks/.envrc` is approved. Always use its direnv environment:

```bash
cd /home/chen-lab/isaac/lab-workspace/isaac-tasks
export ISAAC_TASKS_FR3_USD=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data/generated_assets/fr3v21_bare_flange_live_v1/fr3v2.usda

direnv exec . python scripts/list_envs.py
direnv exec . python -m pytest tests --import-mode=importlib -q

direnv exec . python -u scripts/run_visual_reach_expert.py \
  --contract ../robot-learning/configs/contracts/fr3_visual_reach_v1.json \
  --num_envs 2 --episodes 8 --device cpu --viz none
```

For a visible rollout use `--num_envs 1 --viz kit`. `--viz none` still renders
the two sensor cameras; it is headless, **not** camera-free. A GPU/Vulkan-capable
session is required even with CPU physics. For bounded verification images and
JSON, add `--output-dir /tmp/fr3-v1-check-01` (must not already exist). Without
that option the runner writes its report to stdout only. Persistent artifacts
require an explicit mounted, writable `LAB_DATA_ROOT` with adequate free space.

On another host install `contracts` and `source/isaac_tasks` into the selected
compatible Isaac environment, provide the contract JSON with `--contract`, and
set an audited asset path. No Python import requires sibling checkouts.

## Verification and next gate

The runner checks joint order, finite-difference link Jacobian, known physical
increment mapping, orientation anchoring, invalid action rejection, selective
reset isolation, real RGB geometry/type/nonuniformity, initial red-marker
visibility, camera-frame advancement and expert episode success. It saves a
five-step trace for single/clone comparison and runtime/asset/contract receipts.
Selective reset checks require `--num_envs 2` or more.

Verified on this workstation on 2026-10-14 with CPU physics and RTX sensor
rendering (seed 42, pinned Isaac Lab revision in `model.json`):

| Check | Result |
|---|---|
| Offline tests / workspace source checks | 7 tests / 5 checks passed |
| Final two-clone / single runs | 4/4 and 2/2 expert episodes succeeded |
| Terminal position errors | Approximately 1.46–1.51 cm, below the 2 cm criterion |
| Invalid-state failures, timeouts, projection, stale frame counts | Zero in these runs |
| Link-Jacobian finite-difference maximum difference | ≤ 7.45×10⁻⁵ m/rad |
| Single vs clone five-step joint trace maximum difference | 4.77×10⁻⁷ rad |
| Wrist mount position / rotation maximum difference | 0 m / < 5×10⁻⁷ rad |
| Initial projected marker vs observed red pixels | < 1 pixel in both cameras |
| Selected reset and neighboring state/camera isolation | Passed in the two-clone run |
| Deliberately mismatched camera optical frame | Rejected after Kit initialization, process exit status 1 |

Bounded reports and camera PNGs are in `/tmp/lab-v1-smoke-11` and
`/tmp/lab-v1-smoke-12`; these are temporary verification artifacts, not a
dataset. The runner owns `AppLauncher` shutdown to preserve nonzero failure
status rather than allowing Kit's default fast shutdown to replace it with zero.

This is a bounded integration test, not exhaustive workspace or image-visibility
qualification. Errors are retained from the terminal physics state before
Isaac Lab's automatic reset; returned observations belong to the next episode.

V2's local process bridge is now implemented with episode/step/time metadata,
queue invalidation on reset and validated mock action chunks. See
[V2 endpoint](v2_local_sim.md). Adapted VLA rollouts and dataset collection remain
subsequent work.
