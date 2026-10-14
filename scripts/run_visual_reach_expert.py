"""Finite V1 IK expert verification, not a pretrained VLA rollout or hardware test."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import gymnasium as gym
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lab-VisualReach-FR3-v0")
parser.add_argument("--contract", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=4, help="Episodes per environment")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, help="New directory on LAB_DATA_ROOT, or /tmp for bounded smoke artifacts")
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
import isaac_tasks.tasks  # noqa: E402,F401
_verification_app_launcher = None


def output_directory():
    if args.output_dir is None:
        return None
    path = args.output_dir.expanduser().resolve()
    if path.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Generated artifacts may not be written inside the source repository")
    root = Path(os.environ["LAB_DATA_ROOT"]).resolve() if os.environ.get("LAB_DATA_ROOT") else None
    if not path.is_relative_to(Path("/tmp")):
        if root is None or not path.is_relative_to(root) or not any(os.path.ismount(p) for p in (root, *root.parents) if str(p) != "/"):
            raise ValueError("Explicit mounted LAB_DATA_ROOT required for persistent artifacts")
    parent = next(p for p in path.parents if p.exists())
    if not os.access(parent, os.W_OK) or shutil.disk_usage(parent).free < 64*1024**2:
        raise RuntimeError("Output filesystem unavailable or less than 64 MiB free")
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.mkdir(parents=True)
    return path


def main():
    global _verification_app_launcher
    if not 1 <= args.num_envs <= 4 or not 1 <= args.episodes <= 32:
        raise ValueError("V1 smoke supports 1..4 environments and 1..32 episodes per environment")
    from contracts.policies import PolicyIOSpec
    spec = PolicyIOSpec.from_json(args.contract.read_text())
    cfg, _ = resolve_task_config(args.task, "")
    output = output_directory()
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    if args.device is not None:
        cfg.sim.device = args.device
    # Own Kit lifetime so fast shutdown preserves the verification exit code.
    # The generic launch_simulation context closes Kit with its default status 0.
    from isaaclab.app import AppLauncher
    args.enable_cameras = True
    _verification_app_launcher = AppLauncher(args)
    cfg.sim.device = _verification_app_launcher.device
    with launch_simulation(cfg, args):
        import torch
        from isaac_tasks.bridges.visual_reach import VisualReachBridge
        from isaac_tasks.controllers.visual_reach import tcp_pose_base, expert_increment
        from isaac_tasks.assets.fr3.visual_reach import MODEL
        from isaac_tasks.controllers.visual_reach_checks import verify_fk_and_clone_parity
        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            obs, _ = env.reset(seed=args.seed)
            bridge = VisualReachBridge(env, spec)
            action = env.action_manager.get_term("cartesian_increment")
            diagnostics, checks = verify_fk_and_clone_parity(env)
            obs, _ = env.reset(seed=args.seed)
            # Rejection must precede mutation of the action term.
            before = action.raw_actions.clone()
            for label, value in (("oversized", torch.full_like(before, 0.006)), ("nonfinite", torch.full_like(before, float("nan")))):
                try:
                    action.process_actions(value)
                except ValueError:
                    checks[label + "_rejected"] = torch.equal(before, action.raw_actions)
                else:
                    checks[label + "_rejected"] = False
            p0, r0 = tcp_pose_base(env)
            delta = torch.zeros_like(before)
            delta[:, 2] = 0.001
            action.process_actions(delta)
            checks["measured_tcp_plus_physical_delta"] = bool(torch.allclose(action.target[:, :3], p0+delta, atol=1e-7))
            checks["episode_initial_orientation"] = bool(torch.allclose(action.anchor, r0, atol=1e-7))
            if env.num_envs > 1:
                neighbor = (action.raw_actions[1:].clone(), action.anchor[1:].clone(), env.command_manager.get_command("target")[1:].clone())
                neighbor_q = env.scene["robot"].data.joint_pos.torch[1:].clone()
                neighbor_frames = bridge.frame_counters()[1:]
                env.reach_streak[0] = 3
                env._reset_idx(torch.tensor([0], device=env.device))
                checks["selective_reset"] = all(torch.equal(a, b) for a, b in zip(neighbor,
                    (action.raw_actions[1:], action.anchor[1:], env.command_manager.get_command("target")[1:])))
                checks["selective_reset_neighbor_state_and_cameras"] = (
                    torch.equal(neighbor_q, env.scene["robot"].data.joint_pos.torch[1:])
                    and torch.equal(neighbor_frames, bridge.frame_counters()[1:]))
                checks["selected_action_and_streak_reset"] = (
                    bool(action.needs_anchor[0]) and not bool(action.raw_actions[0].any()) and int(env.reach_streak[0]) == 0)
            obs, _ = env.reset(seed=args.seed)
            features = bridge.policy_features(obs)
            initial_camera_metadata = {key: {field: value.cpu().tolist() for field, value in data.items()}
                for key, data in bridge.camera_metadata().items()}
            checks["privileged_target_not_in_policy"] = set(obs["policy"]) == {"joint_positions", "overhead", "wrist"}
            checks["rgb_nonuniform"] = all(float(features[key].float().std()) > 1 for key in bridge.cameras)
            overhead = features["observation.images.overhead"].float()
            red = (overhead[..., 0] > 80) & (overhead[..., 0] > 1.5*overhead[..., 1]) & (overhead[..., 0] > 1.5*overhead[..., 2])
            checks["red_marker_visible_overhead"] = bool((red.sum(dim=(1, 2)) >= 8).all())
            wrist = features["observation.images.wrist"].float()
            wrist_red = (wrist[..., 0] > 80) & (wrist[..., 0] > 1.5*wrist[..., 1]) & (wrist[..., 0] > 1.5*wrist[..., 2])
            checks["red_marker_visible_wrist"] = bool((wrist_red.sum(dim=(1, 2)) >= 8).all())
            marker_pixel_errors = {key: bridge.marker_pixel_error(features, key).cpu().tolist() for key in bridge.cameras}
            checks["rgb_matches_camera_extrinsics"] = all(max(errors) < 12 for errors in marker_pixel_errors.values())
            if output:
                from PIL import Image
                for key, name in bridge.cameras.items():
                    Image.fromarray(features[key][0].cpu().numpy()).save(output/f"initial_{name}.png")
            counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            results, stale_frames, projection_steps = [], 0, 0
            max_mount_position_error, max_mount_rotation_error = 0.0, 0.0
            saved_approach = False
            previous_frames = bridge.frame_counters()
            for step in range(args.episodes * env.max_episode_length):
                pos, _ = tcp_pose_base(env)
                increments = expert_increment(pos, env.command_manager.get_command("target"))
                obs, _, terminated, truncated, _ = env.step(increments)
                bridge.policy_features(obs)
                mount_pos_error, mount_rot_error = bridge.wrist_mount_error()
                max_mount_position_error = max(max_mount_position_error, float(mount_pos_error.max()))
                max_mount_rotation_error = max(max_mount_rotation_error, float(mount_rot_error.max()))
                frames = bridge.frame_counters()
                done = terminated | truncated
                if output and not saved_approach and not bool(done[0]) and float(env.reach_terminal_error[0]) < 0.025:
                    for name in bridge.cameras.values():
                        Image.fromarray(obs["policy"][name][0].cpu().numpy()).save(output/f"approach_{name}.png")
                    saved_approach = True
                stale_frames += int(((frames <= previous_frames).any(dim=-1) & ~done).sum())
                projection_steps += int(action.projection.sum())
                previous_frames = frames
                # Bounded per-episode reporting is outside the servo hot path.
                for index in done.nonzero().flatten().cpu().tolist():
                    if counts[index] < args.episodes:
                        results.append({"env": index, "success": bool(env.termination_manager.get_term("success")[index]),
                            "invalid_state": bool(env.termination_manager.get_term("invalid_state")[index]),
                            "timeout": bool(truncated[index]), "terminal_error_m": float(env.reach_terminal_error[index])})
                        counts[index] += 1
                if bool((counts >= args.episodes).all()):
                    break
            checks["fresh_camera_frames"] = stale_frames == 0
            checks["wrist_tracks_measured_flange"] = max_mount_position_error < 1e-5 and max_mount_rotation_error < 1e-4
            checks["expert_all_success"] = len(results) == args.episodes*env.num_envs and all(r["success"] and not r["invalid_state"] for r in results)
            report = {"passed": all(checks.values()), "checks": checks, "episodes": results,
                "seed": args.seed, "policy_hz": 1/env.step_dt, "physics_dt_s": cfg.sim.dt,
                "stale_frame_env_steps": stale_frames, "joint_projection_env_steps": projection_steps,
                "diagnostics": diagnostics,
                "wrist_mount_max_position_error_m": max_mount_position_error,
                "wrist_mount_max_rotation_error_rad": max_mount_rotation_error,
                "initial_camera_metadata": initial_camera_metadata,
                "initial_marker_pixel_error": marker_pixel_errors,
                "runtime": {"python": sys.version, **{name: importlib.metadata.version(name)
                    for name in ("isaaclab", "isaacsim", "torch", "contracts", "isaac-tasks")}},
                "asset": MODEL, "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest()}
            if output:
                (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(report, indent=2), flush=True)
            return 0 if report["passed"] else 1
        finally:
            env.close()
    return 1  # Launcher must not suppress errors into a successful exit.


if __name__ == "__main__":
    verification_exit_code = 1
    try:
        verification_exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        if _verification_app_launcher is not None:
            _verification_app_launcher.app.close(exit_code=verification_exit_code)
    raise SystemExit(verification_exit_code)
