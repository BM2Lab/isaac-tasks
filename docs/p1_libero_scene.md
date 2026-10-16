# P1 — native Panda / LIBERO Object scene qualification

P1 implements a manager-based **simulation qualification scene**, not a VLA
benchmark yet. It imports the pinned LIBERO Object task 0 (alphabet soup into
basket): Panda, gripper, room/support surface and all seven free objects. It does
not substitute FR3 assets or import/change historical franka-rl or upstream IsaacLab.

## Implementation

| File | Responsibility |
| --- | --- |
| `assets/panda/libero_scene.py` | Validate native receipts, decode binary MSH to OBJ, split MJCF, stage resets |
| `scripts/prepare_libero_scene.py` | Isaac conversion, static/massless/friction corrections, dependency hashes |
| `controllers/libero_osc_action.py` | Native 7D OSC/gripper action, TCP/Jacobian/mass/bias adapter, selective reset |
| `tasks/manager_based/libero_object/object_env_cfg.py` | `make_config`: scene, actions, diagnostic observations and reset event |
| `scripts/validate_libero_scene.py` | Native probes, hold/step/settling, fingers, support contact, clone/single/reset checks |
| `tests/test_libero_scene.py` | Portable mesh, receipt, component and massless-frame regressions |

Package paths are relative to `source/isaac_tasks/isaac_tasks/`. The native
exporter in robot-learning also captures settled body poses, joint states,
passive parameters and three same-state FK/dynamics probes, including moving
joints. Re-export old P0-only references. No learning or MuJoCo imports enter the
simulation adapter.

The scene is **not Gym-registered as a usable SmolVLA task**. Observations are
diagnostic joint position/velocity (18D), not native 8D state and images.
Camera/image matching, success predicate, scripted grasp/placement and pretrained
policy bridging remain for the next gate.

## Conventions and approximations

- Physics 500 Hz, policy actions 20 Hz. First six normalized inputs retain P0
  native-world translation/rotation scaling and orientation-goal latching.
  All seven arm joints participate in OSC, unlike the FR3 6D policy.
- Grip-site pose and point-shifted Jacobian are reconstructed from
  `robot0_link7` using exported transforms. Clone origins are removed from TCP
  positions, not vector axes. Float32 engine data enter float64 P0 algebra.
- Bias includes gravity **and** Coriolis/centrifugal terms. PhysX's queried mass
  omits rotor armature: the adapter adds its diagonal; armature is also set in
  the solver. Zero implicit arm gains avoid hidden PD control.
- Gripper -1 opens, +1 closes, zero retains the integrated target. Each policy
  step adds $0.01[-1,1]\operatorname{sign}(a_g)$, clips to [-1,1], then maps to
  native finger ranges [0,0.04] / [-0.04,0] m. Motor stiffness is 1000 N/m,
  motor limit 20 N; passive damping is separately 100 N s/m.
- Raw OSC, motor clipping and passive torque remain distinct. No hidden IK,
  filtering, planner or second gravity compensation is added.

Importer corrections live **only in the generated cache**:

1. Convert unsupported legacy binary `.msh` to OBJ; hash originals and outputs.
2. Remove rigid-body API from static room/floor. Use convex collision hulls for
   dynamic meshes; preserve primitive collision geometry.
3. Omit massless leaf/site frames as rigid links, avoiding fictitious default
   mass; recover TCP from metadata. The world-fixed base receives positive dummy
   mass/inertia for valid USD physics, without affecting downstream arm M/J/b.
4. Zero the imported load-dependent PhysX joint-friction coefficient: native
   `frictionloss` is an absolute effort. Use native damping and a **kinetic
   Coulomb sign approximation**, not MuJoCo's solver-level static friction.

Contact/solver/material semantics are not identical across engines. P1 does not
establish identical trajectories, grasping success, benchmark scores, images or
hardware safety. GPU PhysX closed-loop behavior is unqualified; P0 algebra is
independently CPU/CUDA tested.

## Verified results

Verified 2026-10-15: Isaac Sim 6.0.1.0, isaaclab distribution 6.1.14,
Torch 2.10.0+cu128, CPU PhysX, seed 42. Native runtime/asset pins remain P0 pins.
Two clones and a fresh single-env scene passed. Native probes compare identical
q/dq, not different trajectories.

| Measurement | Observed |
| --- | ---: |
| Initial TCP maximum position error | 2.04e-7 m |
| Initial rotation-matrix maximum absolute error | 2.25e-7 |
| Maximum Jacobian error across three probes | 3.73e-7 |
| Maximum mass-matrix error across three probes | 2.15e-6 kg m² |
| Maximum bias-torque error across three probes | 1.15e-5 Nm |
| Held Cartesian goal terminal position error | 0.53 mm |
| Clone final arm-joint difference | 1.87e-6 rad |
| Fresh single versus clone arm-joint difference | 5.91e-6 rad |

Gripper open/close, finite state/torque, soup support contact and selective reset
passed. Support means soup stays on the support surface, not successful grasping.
The settling check sets a goal once and holds it for 500 physics ticks, rather
than reanchoring its position every policy step.

Temporary workstation artifacts (not a persistent dataset): reference
`/tmp/lab-libero-p1-reference-02`; cache `/tmp/libero-p1-cache-07/scene.json`;
reports `/tmp/libero-p1-check-07/report.json` and
`/tmp/libero-p1-single-07/report.json`.

## Commands

First re-export in robot-learning's isolated LIBERO environment; see that repo's
`docs/libero_reference_export.md`. Do not install that stack into Isaac Python.
From isaac-tasks, set explicit `ISAAC_PYTHON`, `NATIVE_REFERENCE` and
`NEW_CONVERSION_CACHE` paths:

```bash
direnv exec . "$ISAAC_PYTHON" scripts/prepare_libero_scene.py \
  --reference-dir "$NATIVE_REFERENCE" --output-dir "$NEW_CONVERSION_CACHE" \
  --device cpu --viz none

direnv exec . "$ISAAC_PYTHON" scripts/validate_libero_scene.py \
  --scene-manifest "$NEW_CONVERSION_CACHE/scene.json" \
  --output-dir /tmp/libero-p1-two --num_envs 2 --device cpu --viz none

direnv exec . "$ISAAC_PYTHON" scripts/validate_libero_scene.py \
  --scene-manifest "$NEW_CONVERSION_CACHE/scene.json" \
  --output-dir /tmp/libero-p1-one --num_envs 1 --device cpu --viz none \
  --compare-report /tmp/libero-p1-two/report.json
```

Cache parent must exist on a local conversion filesystem with 2 GiB free; the
cache itself must be new. Results require new `/tmp` directories for bounded
smoke, or an explicit mounted `LAB_DATA_ROOT`. For visible single-env diagnostics
use `--viz kit`, omit `--compare-report`, and choose a new output directory.
This is a finite qualification sequence, not a live policy session. Generated
assets/results stay out of Git; converted dependency hashes are checked at load.
