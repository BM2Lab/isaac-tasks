"""P2 one-env camera/contract endpoint or bounded visual qualification."""
import argparse
import json
import os
from pathlib import Path
import shutil
import traceback

from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene-manifest", type=Path, required=True)
parser.add_argument("--visual-reference", type=Path, required=True)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--socket", type=Path)
mode.add_argument("--check-output", type=Path)
parser.add_argument("--max-steps", type=int, default=280)
parser.add_argument("--idle-timeout", type=float, default=120)
parser.add_argument("--max-requests", type=int, default=1000)
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args, _ = setup_preset_cli(parser)
launcher = None


def verify(session, output):
    import numpy as np
    import torch
    from PIL import Image
    from contracts.libero import LiberoActionChunk
    from isaac_tasks.bridges.libero_visual import synchronize_cameras
    output = output.expanduser().resolve()
    if not output.is_relative_to(Path("/tmp")):
        root = Path(os.environ.get("LAB_DATA_ROOT", "/not-configured")).resolve()
        if not output.is_relative_to(root) or not any(os.path.ismount(p) for p in (root, *root.parents) if str(p) != "/"):
            raise ValueError("Use /tmp for bounded smoke or an explicit mounted LAB_DATA_ROOT")
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 64*1024**2:
        raise RuntimeError("Existing output parent with 64 MiB free required")
    output.mkdir(exist_ok=False)
    first = session.observe()
    checks = {"observation_stable": session.observe() == first}
    difference = float(np.abs(np.asarray(first["state"])-session.reference["visual"]["state_same_state"]).max())
    cached_difference = float(np.abs(np.asarray(first["state"])-session.reference["visual"]["state"]).max())
    checks["state_native_parity"] = difference < 2e-5
    camera_metrics = {}
    for source, short in (("agentview", "agentview"), ("robot0_eye_in_hand", "eye_in_hand")):
        sensor = session.env.scene[short]
        camera = session.reference["visual"]["cameras"][source]
        image = session.observations["policy"]["image" if short == "agentview" else "image2"][0].cpu().numpy()
        Image.fromarray(image).save(output/f"isaac_{short}.png")
        shutil.copyfile(args.visual_reference/camera["image"], output/f"native_{short}.png")
        actual_p = sensor.data.pos_w.torch[0].cpu().numpy()
        from isaaclab.utils.math import matrix_from_quat
        actual_r = matrix_from_quat(sensor.data.quat_w_opengl.torch)[0].cpu().numpy()
        intrinsic = sensor.data.intrinsic_matrices.torch[0].cpu().numpy()
        focal = 128/np.tan(np.deg2rad(camera["fovy_degrees"])/2)
        metrics = {"position_error_m": float(np.abs(actual_p-camera["position_world"]).max()),
                   "rotation_matrix_error": float(np.abs(actual_r-camera["rotation_world_opengl"]).max()),
                   "focal_error_px": float(max(abs(intrinsic[0, 0]-focal), abs(intrinsic[1, 1]-focal))),
                   "rgb_std": float(image.std()), "intrinsics": intrinsic.tolist()}
        camera_metrics[short] = metrics
        checks[short+"_geometry"] = metrics["position_error_m"] < 2e-5 and metrics["rotation_matrix_error"] < 2e-5 and metrics["focal_error_px"] < .01
        checks[short+"_rgb"] = image.dtype == np.uint8 and image.std() > 5
    context = session.context
    try:
        session.advance(context.to_dict())
    except ValueError:
        checks["empty_queue_no_step"] = session.context == context
    else:
        checks["empty_queue_no_step"] = False
    # Expiry is checked at each execution, not merely on submit.
    row = (0.,)*7
    chunk = LiberoActionChunk(context, session.spec.action, (row, row), context.timestamp_ns+1)
    session.submit(chunk.to_dict())
    session.advance(context.to_dict())
    current = session.context
    try:
        session.advance(current.to_dict())
    except ValueError:
        checks["execution_expiry_no_step"] = session.context == current and len(session.queue) == 0
    else:
        checks["execution_expiry_no_step"] = False
    first_frames = [i["frame_index"] for i in first["images"]]
    checks["fresh_frames"] = all(i["frame_index"] > n for i, n in zip(session.observe()["images"], first_frames))
    before = session.context
    session.submit(LiberoActionChunk(before, session.spec.action, (row,), before.timestamp_ns+100_000_000).to_dict())
    session.reset()
    checks["reset_queue_identity"] = len(session.queue) == 0 and session.context.episode_id != before.episode_id and session.context.step_index == 0
    before = session.context
    try:
        session.advance(context.to_dict())
    except ValueError:
        checks["stale_advance_no_step"] = session.context == before
    else:
        checks["stale_advance_no_step"] = False
    # Finite neutral horizon: preserve terminal capture, no implicit reset or
    # unimplemented success claim. Keep this test bounded to three steps.
    session.max_steps = 3
    terminal = None
    for _ in range(3):
        context = session.context
        session.submit(LiberoActionChunk(context, session.spec.action, (row,), context.timestamp_ns+100_000_000).to_dict())
        result = session.advance(context.to_dict())
        terminal = result["terminal"]
    checks["terminal_capture"] = terminal == {"reason": "horizon", "success": None} and session.context.step_index == 3
    # Camera mount must also follow a nonzero measured wrist pose, without
    # advancing physics while reading observations.
    action = session.env.action_manager.get_term("osc")
    robot = session.env.scene["robot"]
    q = robot.data.joint_pos.torch[:, action.arm_ids].clone()
    q[:, 0] += .05
    robot.write_joint_position_to_sim_index(position=q, joint_ids=action.arm_ids)
    session.env.sim.forward()
    session.env.scene.update(session.env.physics_dt)
    synchronize_cameras(session.env)
    from isaaclab.utils.math import matrix_from_quat
    link_r = matrix_from_quat(robot.data.body_link_quat_w.torch[:, action.body])
    sensor = session.env.scene["eye_in_hand"]
    expected = robot.data.body_link_pos_w.torch[:, action.body]+(link_r@q.new_tensor(sensor.cfg.mount_position)[None, :, None]).squeeze(-1)
    mount_error = float((sensor.data.pos_w.torch-expected).abs().max())
    checks["measured_mount_sync"] = mount_error < 2e-5
    checks = {key: bool(value) for key, value in checks.items()}
    report = {"passed": all(checks.values()), "checks": checks, "state_native_max_abs": difference,
              "cached_native_reset_observation_max_abs": cached_difference,
              "camera_metrics": camera_metrics, "mount_position_error_m": mount_error,
              "receipt": session.receipt(), "scope": "P2 bridge/cameras, no model rollout or task-success claim"}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


def main():
    global launcher
    if not 1 <= args.max_requests <= 10000 or not 1 <= args.idle_timeout <= 300:
        raise ValueError("Invalid IPC request/idle limits")
    from isaaclab.app import AppLauncher
    args.enable_cameras = True
    launcher = AppLauncher(args)
    from isaaclab.envs import ManagerBasedEnv
    from isaac_tasks.tasks.manager_based.libero_object.visual_env_cfg import make_visual_config
    from isaac_tasks.bridges.libero_session import LocalLiberoSession
    cfg = make_visual_config(args.scene_manifest, args.visual_reference, device=args.device or "cpu")
    cfg.sim.device = launcher.device
    with launch_simulation(cfg, args):
        env = ManagerBasedEnv(cfg=cfg)
        try:
            session = LocalLiberoSession(env, args.visual_reference, max_steps=args.max_steps)
            if args.check_output:
                return verify(session, args.check_output)
            from isaac_tasks.bridges.local_server import serve
            serve(session, args.socket, args.idle_timeout, args.max_requests)
            return 0
        finally:
            env.close()
    return 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        traceback.print_exc()
    finally:
        if launcher is not None:
            launcher.app.close(exit_code=code)
    raise SystemExit(code)
