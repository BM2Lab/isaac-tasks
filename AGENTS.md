# isaac-tasks contributor instructions

## Isaac Lab architecture

This is an external Isaac Lab extension. The installable package lives at
`source/isaac_tasks`, with extension metadata in `config/extension.toml` and
Python code in `isaac_tasks`. Follow the generated manager-based workflow:

- `tasks/manager_based/<task>/<task>_env_cfg.py`: scene and action, observation,
  command, event, reward, termination and curriculum configuration groups.
- Task-local `mdp/`: custom vectorized terms; reuse official Isaac Lab terms
  where compatible. Shared terms belong in `manager_based/common_mdp/`.
- `agents/`: RL runner configuration; `config/<embodiment>/`: robot-specific
  task specializations and registration boundaries.
- `assets/`, `controllers/`, `bridges/`: reusable model configuration, simulated
  adapters and contract/recording integration, not application workflow logic.

Register Gym environments only when their configurations are complete and tested.
Keep scene, joint/TCP mapping, controller/action semantics and success criteria
explicit. No placeholder environment should appear as a usable task.

Depend on the lightweight `contracts` distribution, not the full ML orchestrator.
Never import real Franky/FCI/CAN drivers or implement physical robot connections
here. Application-specific rewards and workflows remain application-owned.

## Runtime and verification

Use the host's pinned Isaac Sim/Lab environment and launcher/direnv setup.
Do not hard-code a workstation interpreter, asset cache or sibling repo path.
Do not upgrade the simulator stack implicitly; record versions and asset provenance.
Keep runtime imports compatible with the selected Isaac Lab release.

Run offline tests in that environment:

```bash
python -m pytest tests --import-mode=importlib
python scripts/list_envs.py
```

When adding runnable tasks, test headless scene creation, joint order/FK/TCP,
finite observations/actions, selective resets, clone/single parity and physical
parameter application before reward tuning. Keep physics stepping, batched
actions/state/rewards and observations device-resident. Avoid per-environment
Python loops, CPU synchronization and file I/O in the hot path. Training must
not require GUI/cameras unless explicitly part of the task.

For extracted code, record source revisions and retain baseline regression
fixtures; do not change control laws during an unreviewed migration. Distinguish
simulator guards/model approximations from actual hardware safety guarantees.

## Storage and current status

The generic reach configuration is intentionally incomplete and unregistered.
The FR3 visual-reaching specialization is complete; see `docs/v1_visual_reach.md`
for its physical action contract and simulation-only verification. Its simplified
IK/implicit-PD controller is not hardware-faithful. Other embodiments and extracted
hardware-faithful controllers require separate implementation and validation.
Keep large assets/generated USDs, checkpoints, videos and experiment logs outside
Git; retain lightweight model provenance, configuration and fixtures. Use explicit
artifact roots with availability/capacity checks, never an implicit repo fallback.
No live hardware tests or actuation are authorized by simulator tests or CI.

V2's local socket endpoint is simulation-only and single-environment. Preserve
exact contract handshaking, episode/capture identity, per-execution expiry checks,
reset queue invalidation and private socket lifecycle. Waiting for a client must
not advance physics. Never import the learning/ML stack to run the endpoint;
see `docs/v2_local_sim.md` for protocol and verification.
