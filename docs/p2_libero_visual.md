# P2 — native visual contract and paused local bridge

P2 adds calibrated cameras, an explicit native observation/action contract and
a simulation-only endpoint on the [P1 scene](p1_libero_scene.md). It does **not**
run learned actions, register a complete LIBERO benchmark, implement grasp/place
success, or authorize hardware. P3 must qualify actual policy rollout and task
success. P1 controller gains, action semantics and physics cadence are unchanged.

## Contract

`contracts.libero.LiberoIOSpec` has ID `libero_object0_osc_visual_v1`, distinct
from the FR3 V0 policy specification. It is standard-library-only.

| Feature | Meaning |
| --- | --- |
| `observation.state[0:3]` | Grip-site position in native LIBERO world, meters |
| `observation.state[3:6]` | **robot0_right_hand body** quaternion converted by LeRobot's rotation-vector formula, radians |
| `observation.state[6:8]` | Left/right finger qpos, meters; native signed joint conventions |
| `observation.images.image` | Agent view, 256x256 uint8 RGB HWC |
| `observation.images.image2` | Eye-in-hand view, same format |
| `task` | Native language instruction; no privileged object coordinates |
| Action, 7D | Six normalized OSC pose inputs plus signed gripper; NOT physical meters |

Native position and orientation observations refer to **different frames**.
The OSC controller itself uses grip-site orientation. Native quaternion signs
are preserved by composing the exported body/joint FK chain; forcing w>=0 or
using the controller's rotation matrix produces a different policy state.

MuJoCo raw RGB is bottom-up. LeRobot's LIBERO environment processor then flips
both H and W. Isaac RGB is already top-down, so the equivalent transform from
Isaac is **one horizontal flip**. Packets are already in processor-output
orientation: the learning client must not apply the LIBERO environment flip
again. The checkpoint's own policy normalization/tokenization remains necessary.

The camera factory uses native measured OpenGL extrinsics and fovy. The wrist
camera is an independent prim synchronized from measured link-7 pose before
rendering, avoiding stale child transforms. State/RGB share one capture context.

## Files

- `assets/panda/libero_visual_assets.py`: source texture/material and visibility
  repairs; render-only floor quad for native planar texture projection.
- `tasks/manager_based/libero_object/visual_env_cfg.py`: P2 scene/camera factory.
- `bridges/libero_visual.py`: native state, quaternion branch and image mapping.
- `bridges/libero_session.py`: `LocalLiberoSession`, explicit reset/horizon and
  episode-bound queue execution. No ML imports.
- `scripts/serve_libero.py`: endpoint or bounded camera/state qualification.
- robot-learning: `contracts/libero.py`, `learning/integrations/isaaclab/libero.py`,
  and `scripts/check_libero_bridge.py` provide the typed contract and client.

The existing private socket framing, permissions, exact handshake, per-action
expiry and reset invalidation are reused. Waiting for a client never steps
physics. Repeated observes return the same capture. Horizon completion preserves
the terminal observation without automatic reset; `success` is **null**, not
false or a fabricated benchmark score. Runtime failures fault/close the session.

## Asset repairs and remaining visual gap

Rebuild assets with `prepare_libero_scene.py --native-visuals`. P2 rejects older
unrepaired caches. The importer left shader references outside the default prim
and PNG references in a deleted temporary directory. P2 authors local shading
networks using original pinned textures, retaining snapshot `.png` filenames
(resolving them to extensionless Hugging Face blobs breaks RTX texture loading).
Texture hashes are checked at load. Collision debug geometry and sites are hidden
from rendering, not disabled in physics. A visual-only textured quad supplements
the original plane collider, which has no UVs.

Native and Isaac camera framing match, but **images are not pixel-identical**:
lighting, shading, texture projection and rasterization differ. RTX dome intensity
is explicitly 300; this is exposure calibration, not MuJoCo lighting parity.
Paired images were visually inspected. Remaining visual domain shift is a risk
for pretrained-policy transfer and must not be mistaken for proven VLA success.

## Verification (2026-10-15)

CPU PhysX, RTX rendering, fixed Object-0 fixture:

- Fresh same-state 8D feature error: **2.02e-7 maximum absolute**.
- Cached native reset observation differs by 1.45e-4: native observables may be
  slightly older than the current physical state. Both references are retained.
- Camera position error <=2.11e-7 m, rotation-matrix error <=2.20e-7,
  focal-length error <=1.84e-5 pixels; measured wrist mount sync passed.
- Both nonblank RGB views, fresh frame indices, repeat-observe identity,
  empty/expired queue rejection without stepping, reset invalidation, stale
  advance rejection and terminal capture passed.
- Actual isolated learning-client/socket test passed three neutral policy steps,
  reset invalidation and clean close: **zero model actions**.

Temporary artifacts: `/tmp/libero-p2-native-03`, cache
`/tmp/libero-p2-cache-04/scene.json`, paired images/report
`/tmp/libero-p2-visual-final`, IPC receipt/images `/tmp/libero-p2-client-01`.
These are bounded development checks, not a persistent dataset or score.
The complete P1 two-clone physics suite also passed on the repaired P2 assets
(`/tmp/libero-p2-physics-regression`). Offline suites passed 35 isaac-tasks tests
and 110 contracts/learning tests, including existing FR3 endpoint regression tests.

## Commands

Re-export a P2 reference in robot-learning's isolated LIBERO environment using
`scripts/export_libero_reference.py`; see its reference-export documentation.
From isaac-tasks with explicit paths and its direnv:

```bash
direnv exec . "$ISAAC_PYTHON" scripts/prepare_libero_scene.py \
  --reference-dir "$NATIVE_REFERENCE" --output-dir "$NEW_CACHE" \
  --native-visuals --device cpu --viz none

direnv exec . "$ISAAC_PYTHON" scripts/serve_libero.py \
  --scene-manifest "$NEW_CACHE/scene.json" --visual-reference "$NATIVE_REFERENCE" \
  --check-output /tmp/libero-p2-check --device cpu --viz none
```

For an endpoint, replace `--check-output` with `--socket "$PRIVATE_SOCKET"`
and optionally `--max-steps 3` for a terminal-horizon smoke. Socket parent must
be private, owned by the user, mode 0700. Then use robot-learning's isolated
learning Python and direnv to run `scripts/check_libero_bridge.py --socket ...
--output-dir /tmp/libero-p2-client`. No policy weights are loaded in either check.
Use fresh output/cache directories; generated data stay outside Git. `--viz kit`
is optional for a visible scene, not required for offscreen camera rendering.
