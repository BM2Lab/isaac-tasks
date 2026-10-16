# isaac-tasks

External Isaac Lab extension, following the manager-based project layout used
by franka-rl. Install source/isaac_tasks (not the repository root) into the pinned
Isaac environment alongside the lightweight contracts package.

source/isaac_tasks/config/extension.toml describes the extension.
source/isaac_tasks/isaac_tasks/tasks/manager_based/reach/ owns:
- reach_env_cfg.py: scene, actions, observations, commands, events, rewards,
  terminations, curriculum and the ManagerBasedRLEnvCfg.
- mdp/: task-local terms, reusing Isaac Lab MDP implementations where appropriate.
- agents/: PPO/RL configuration placeholders.
- config/{fr3,panda,openarm}/: embodiment-specific config and registration boundaries.

Shared MDP terms belong in tasks/manager_based/common_mdp/. Assets, controller
adapters and bridges remain outside individual task folders.

Importing isaac_tasks discovers task packages like the generated template.
`Lab-VisualReach-FR3-v0` is the first complete, camera-equipped V1 environment.
See [V1 setup, contract and verification](docs/v1_visual_reach.md).
The generic ReachEnvCfg remains intentionally non-runnable; Panda and OpenArm
specializations are still placeholders. No assets are downloaded and the historical
FR3 controller implementation is not modified.
scripts/list_envs.py checks registration without launching a simulator.

Do not move native hardware drivers here. Migration requires source provenance,
FK/joint-map, reset, controller/reference and reward parity against franka-rl.

V2 adds a simulation-only local endpoint with validated action chunks and
episode-aware reset invalidation. See [V2 simulator endpoint](docs/v2_local_sim.md).

V3 adds an opt-in privileged expert label endpoint; dataset writing/export remains
in the separate learning process. See [V3 expert endpoint](docs/v3_demonstrations.md).

P0 adds an offline, batched native-LIBERO OSC reference controller and numerical
parity checks, not a runnable Panda scene yet. See [P0 qualification](docs/p0_libero_osc.md).

P1 adds the imported native Panda/soup/basket manager-based qualification scene,
its OSC/gripper action adapter, and reference checks. See
[P1 setup and limitations](docs/p1_libero_scene.md).

P2 adds native 8D EE/gripper features, two calibrated RGB views and a paused local
endpoint with an isolated learning client. See [P2 visual bridge](docs/p2_libero_visual.md).
Learned rollout, task-success qualification and a registered VLA benchmark remain
separate work; matching camera geometry does not imply pixel-identical rendering.
