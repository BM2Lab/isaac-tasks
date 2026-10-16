# P0 — native LIBERO OSC reference and Torch parity

Implemented 2026-10-15. This is an **offline controller-algebra qualification**,
not an Isaac scene, policy rollout, physics-equivalence or hardware-safety claim.
The first planned scene is LIBERO Object task 0 (soup in basket), using native
Panda/gripper semantics, not the existing FR3 visual-reaching approximation.

## Files and ownership

| Repository / file | Responsibility |
| --- | --- |
| robot-learning: `learning/integrations/libero/reference.py` | Native model/initial-state receipt and robosuite oracle export |
| robot-learning: `scripts/export_libero_reference.py` | Isolated native export entry point; no model inference |
| isaac-tasks: `controllers/libero_osc.py` | `OscState`, `LiberoOscController`, cross-column orientation error |
| isaac-tasks: `controllers/libero_osc_checks.py` | `check_reference` numerical comparison |
| isaac-tasks: `scripts/validate_libero_osc.py` | Offline CPU/CUDA parity CLI, nonzero on mismatch |
| isaac-tasks: `tests/test_libero_osc.py`, `tests/fixtures/libero_osc_reference.json` | Native regression fixture; batch/single, selective reset and compensation tests |

Paths beginning with `controllers/` are relative to
`source/isaac_tasks/isaac_tasks/`. The simulator package does not depend on
`learning`, MuJoCo, robosuite or LeRobot. The native exporter imports its optional
dependencies only after configuring the isolated runtime/cache.

## Exact profile being reproduced

- robosuite **1.4.0**, hf-libero **0.1.4**, MuJoCo **3.8.1**. Immutable asset and
  model pin information comes from robot-learning's `smolvla_libero.json`.
- Six arm action components, clipped to [-1, 1], scaled to translation +/-0.05 m
  and rotation +/-0.5 rad. The seventh native action controls the gripper and is
  **not implemented by this arm-only controller**.
- Native world-frame pose and geometric Jacobian, linear rows before angular.
  Rotation matrices have frame axes in columns. All seven arm joints participate.
- Fixed Cartesian gains 150, damping $2\sqrt{150}$; no interpolation, filtering,
  torque slew limiter or IK. Latched goals are set at 20 Hz; the native model's
  physics timestep is 0.002 s.
- Position goal is reanchored at measured position on every policy action.
  Nonzero rotation is applied on the **left** of measured orientation. Exactly
  zero rotation retains the old orientation goal, rather than reanchoring it.
- Native orientation error is
  $e_R=\frac12\sum_{i=1}^{3}R_{:,i}\times R^*_{:,i}$, not an SO(3) log map.
  Its zero error at an exact 180-degree difference is deliberately preserved.

For $J=[J_p;J_R]$ and mass matrix $M$:

$$
\Lambda=(JM^{-1}J^T)^+,\qquad
N=I-M^{-1}J^T\Lambda J.
$$

The default uncoupled task wrench uses separate
$(J_pM^{-1}J_p^T)^+$ and $(J_RM^{-1}J_R^T)^+$ blocks, while the null-space
projector still uses the **full** $\Lambda$:

$$
\tau=J^T\begin{bmatrix}
\Lambda_p(K_pe_p-K_dv)\\
\Lambda_R(K_pe_R-K_d\omega)
\end{bmatrix}
+b+N^TM\left(10(q_0-q)-2\sqrt{10}\dot q\right).
$$

$b$ is MuJoCo's `qfrc_bias`: gravity plus Coriolis/centrifugal terms.
$q_0$ is the actual native controller's initial joint vector, not an assumed home
or the measured post-settling vector. Raw torque is returned separately from the
final robot torque-limit clipping. Caller must provide consistent measured state,
joint order, bias, Jacobian and SPD mass matrix. Goals must be reset explicitly
at episode boundaries; selective reset leaves neighboring goals/anchors intact.

## Numerical scope and verification

The native pseudoinverse uses NumPy's default relative cutoff $10^{-15}$.
P0 explicitly reproduces it with **float64**; float32 inputs are rejected until
rank handling and acceptable physical behavior are separately qualified. Native
quaternion-to-matrix float32 intermediate rounding is retained approximately.
Do not silently add regularization or change the cutoff and call it parity.

The fixture contains one actual native reset snapshot, plus deterministic action
and synthetic moving/singular/rotation/coupling variants. The oracle invokes
upstream `set_goal` and `run_controller` unchanged, overriding only state
acquisition. It is not nine physical trajectories. Full exports also contain
MJCF, selected initial-state row/digest, body mass/inertia/COM, joint names/ranges,
arm state indices, TCP site, timestep, native controller settings and source hashes.
MJCF references the pinned asset cache; it is not a self-contained asset bundle.

Verified on CPU and RTX 4090 CUDA: **9/9 cases passed**, with
`atol=2e-5`, `rtol=2e-6`. Checks cover goals, raw/clipped torque and null-space
matrix. Largest clipped torque difference was $8.47\times10^{-6}$ Nm. The
deliberately near-singular case amplified goal rounding into about 0.54 Nm raw
torque difference, passing the stated relative tolerance; final clipped torque
matched. This is **not** a uniform raw-torque absolute-error bound.
Tests additionally cover bias, posture torque, clipping, selective reset,
batch/single parity, goal latching and the native 180-degree behavior.

23 isaac-tasks tests and 97 contracts/learning tests passed. No new Gym task is
registered. Historical franka-rl and the IsaacLab checkout were not modified.

From this repository in the pinned Isaac Python environment (no Kit launch):

```bash
python scripts/validate_libero_osc.py \
  --reference tests/fixtures/libero_osc_reference.json
python scripts/validate_libero_osc.py \
  --reference tests/fixtures/libero_osc_reference.json --device cuda:0
python -m pytest tests --import-mode=importlib
```

## Provenance and next gate

The primary law is adapted from installed robosuite 1.4.0 `controllers/osc.py`
and `utils/control_utils.py` / `transform_utils.py`; source hashes are in the
fixture. See [robosuite MIT notice](../THIRD_PARTY_NOTICES.md).
Baseline franka-rl revision `89cf4f40cd8edd39c8f7184ffbe58bab31280f7e`
(`franka_motion/impedance.py`, motion-control `mdp/controllers.py` and
`mdp/actions.py`) informed batched reset/projector and future data-adapter design.
Its log-map error, simpler impedance law, filters and torque processing were
**not** substituted for native OSC. There is no dependency on that checkout.
Isaac Lab's existing OSC is not a drop-in numerical match for this partial
decoupling/null-space/pseudoinverse combination.

P1 must build the Panda/object scene and validate joint/TCP mapping, FK,
Jacobian/mass/bias acquisition and compensation, reset/clone behavior, gripper,
contact and closed-loop references before any SmolVLA rollout. A float32
production profile, if needed, requires its own precision/singularity gate.
