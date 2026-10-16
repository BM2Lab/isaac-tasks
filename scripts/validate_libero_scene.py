"""P1 headless/visible scene qualification; no policy or hardware imports."""
import argparse
import json
import os
from pathlib import Path
import shutil
import traceback
import importlib.metadata
import inspect

from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene-manifest", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--compare-report", type=Path, help="Qualified multi-env report for a fresh single-env parity run")
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args, _ = setup_preset_cli(parser)
launcher = None


def main():
    global launcher
    import torch
    from isaaclab.app import AppLauncher
    if not 1 <= args.num_envs <= 4:
        raise ValueError("Bounded P1 qualification supports 1..4 environments")
    output = args.output_dir.expanduser().resolve()
    if not output.is_relative_to(Path("/tmp")):
        root = Path(os.environ.get("LAB_DATA_ROOT", "/not-configured")).resolve()
        if not output.is_relative_to(root) or not any(os.path.ismount(p) for p in (root, *root.parents) if str(p) != "/"):
            raise ValueError("Use /tmp for bounded smoke or an explicit mounted LAB_DATA_ROOT")
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 64*1024**2:
        raise RuntimeError("Existing output parent with 64 MiB free required")
    output.mkdir(exist_ok=False)
    launcher = AppLauncher(args)
    from isaaclab.envs import ManagerBasedEnv
    from isaac_tasks.tasks.manager_based.libero_object.object_env_cfg import make_config
    from isaac_tasks.assets.panda.libero_scene import digest
    cfg = make_config(args.scene_manifest, num_envs=args.num_envs, device=args.device or "cpu")
    cfg.sim.device = launcher.device
    with launch_simulation(cfg, args):
        env = ManagerBasedEnv(cfg=cfg)
        try:
            env.reset()
            action = env.action_manager.get_term("osc")
            state = action.state()
            native = action.ref["cases"][0]["state"]
            def error(key):
                return float((getattr(state, key)-torch.tensor(native[key], device=env.device)).abs().max())
            metrics = {key+"_native_max_abs": error(key) for key in ("position", "rotation", "jacobian", "mass", "bias")}
            metrics["mass_diagonal"] = state.mass[0].diag().tolist()
            metrics["native_mass_diagonal"] = torch.tensor(native["mass"]).diag().tolist()
            metrics["bias"] = state.bias[0].tolist()
            metrics["native_bias"] = native["bias"]
            metrics["clone_initial_difference"] = float((state.joint_position-state.joint_position[:1]).abs().max())
            checks = {"fk": metrics["position_native_max_abs"] < 2e-5,
                      "tcp_rotation": metrics["rotation_native_max_abs"] < 2e-5,
                      "jacobian": metrics["jacobian_native_max_abs"] < 2e-5,
                      "mass": metrics["mass_native_max_abs"] < 1e-4,
                      "gravity_bias": metrics["bias_native_max_abs"] < 1e-3,
                      "clone_initial": metrics["clone_initial_difference"] < 1e-6}
            probe_results = []
            robot = env.scene["robot"]
            for probe in action.ref["scene"].get("probes", []):
                robot.write_joint_position_to_sim_index(position=torch.tensor(probe["joint_position"],
                    device=env.device).expand(env.num_envs, -1).contiguous(), joint_ids=action.arm_ids)
                robot.write_joint_velocity_to_sim_index(velocity=torch.tensor(probe["joint_velocity"],
                    device=env.device).expand(env.num_envs, -1).contiguous(), joint_ids=action.arm_ids)
                env.sim.forward()
                env.scene.update(env.physics_dt)
                measured = action.state()
                probe_results.append({key: float((getattr(measured, key)-torch.tensor(probe[key], device=env.device)).abs().max())
                    for key in ("position", "rotation", "jacobian", "mass", "bias")})
            checks["same_state_probes"] = bool(probe_results) and all(
                row["position"] < 2e-5 and row["rotation"] < 2e-5 and row["jacobian"] < 2e-5
                and row["mass"] < 1e-4 and row["bias"] < 1e-3 for row in probe_results)
            metrics["same_state_probes_max_abs"] = probe_results
            env.reset()
            state = action.state()
            zero = torch.zeros((env.num_envs, 7), device=env.device)
            start = state.position.clone()
            for _ in range(10):
                env.step(zero)
            metrics["hold_drift_m"] = float((action.state().position-start).norm(dim=-1).max())
            checks["hold"] = metrics["hold_drift_m"] < .005
            start = action.state().position.clone()
            delta = zero.clone()
            delta[:, 0] = .1
            for _ in range(10):
                env.step(delta)
            displacement = action.state().position-start
            metrics["delta_x_m"] = displacement[:, 0].tolist()
            metrics["delta_raw_torque"] = action.last_raw_torque[0].tolist()
            metrics["delta_joint_velocity"] = action.state().joint_velocity[0].tolist()
            checks["positive_x_response"] = bool(((displacement[:, 0] > .005) & (displacement[:, 0] < .03)).all())
            # A one-shot Cartesian step with HELD goals checks actual settling,
            # rather than repeatedly reanchoring position with zero actions.
            action.process_actions(delta)
            for tick in range(500):
                action.apply_actions()
                env.scene.write_data_to_sim()
                env.sim.step(render=tick % 25 == 0)
                env.scene.update(env.physics_dt)
            metrics["latched_goal_position_error_m"] = float((action.state().position-action.controller.goal_position).norm(dim=-1).max())
            checks["latched_goal_settling"] = metrics["latched_goal_position_error_m"] < .002
            # Both fingers must open and then close; no instant position teleport.
            opened = zero.clone()
            opened[:, 6] = -1
            for _ in range(30):
                env.step(opened)
            opening = env.scene["robot"].data.joint_pos.torch[:, action.finger_ids].abs().sum(-1).clone()
            opened[:, 6] = 1
            for _ in range(60):
                env.step(opened)
            closing = env.scene["robot"].data.joint_pos.torch[:, action.finger_ids].abs().sum(-1)
            metrics["finger_open_close_delta_m"] = (opening-closing).tolist()
            checks["gripper"] = bool(((opening-closing) > .003).all())
            soup = env.scene["alphabet_soup_1_main"]
            pose = soup.data.root_pos_w.torch-env.scene.env_origins
            native_z = action.ref["scene"]["body_pose_wxyz"]["alphabet_soup_1_main"][2]
            metrics["soup_height_m"] = pose[:, 2].tolist()
            checks["support_contact"] = bool(((pose[:, 2]-native_z).abs() < .02).all())
            checks["finite"] = bool(torch.isfinite(action.state().joint_position).all() and torch.isfinite(action.last_raw_torque).all())
            metrics["clone_final_joint_difference"] = float((action.state().joint_position-action.state().joint_position[:1]).abs().max())
            checks["clone_final"] = metrics["clone_final_joint_difference"] < 1e-3
            metrics["final_joint_position"] = action.state().joint_position.tolist()
            metrics["final_tcp_position"] = action.state().position.tolist()
            if args.compare_report:
                previous = json.loads(args.compare_report.read_text())
                if (not previous["passed"] or previous["num_envs"] < 2 or env.num_envs != 1
                        or previous["scene_manifest"] != str(args.scene_manifest.resolve())
                        or previous.get("scene_manifest_sha256") != digest(args.scene_manifest)):
                    raise ValueError("Clone/single comparison requires a passed multi-env report from this scene")
                difference = (action.state().joint_position[0].cpu()-torch.tensor(
                    previous["metrics"]["final_joint_position"][0])).abs().max()
                metrics["single_vs_clone_joint_max_abs"] = float(difference)
                checks["single_vs_clone"] = float(difference) < 1e-3
            if env.num_envs > 1:
                before = env.scene["robot"].data.joint_pos.torch[1:].clone()
                before_goal = action.controller.goal_position[1:].clone()
                env.reset(env_ids=torch.tensor([0], device=env.device))
                checks["selective_reset"] = bool(torch.equal(before, env.scene["robot"].data.joint_pos.torch[1:])
                    and torch.equal(before_goal, action.controller.goal_position[1:]))
            report = {"passed": all(checks.values()), "checks": checks, "metrics": metrics,
                      "scene_manifest": str(args.scene_manifest.resolve()), "num_envs": env.num_envs,
                      "scene_manifest_sha256": digest(args.scene_manifest),
                      "runtime": {name: importlib.metadata.version(name) for name in ("isaacsim", "isaaclab", "torch")},
                      "source_sha256": {Path(p).name: digest(p) for p in (__file__, inspect.getfile(type(action)),
                          inspect.getfile(type(action.controller)), inspect.getfile(make_config))},
                      "device": str(env.device),
                      "physics_dt": env.physics_dt, "policy_dt": env.step_dt,
                      "scope": "P1 reference scene only, not a pretrained rollout or hardware qualification"}
            (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(report, indent=2), flush=True)
            return 0 if report["passed"] else 1
        finally:
            env.close()
    return 1


if __name__ == "__main__":
    status = 1
    try:
        status = main()
    except Exception:
        traceback.print_exc()
    finally:
        if launcher is not None:
            launcher.app.close(exit_code=status)
    raise SystemExit(status)
