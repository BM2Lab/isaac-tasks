"""V2 private local simulator endpoint; one environment, no ML or hardware runtime."""
import argparse
from pathlib import Path
import sys
import traceback
import gymnasium as gym
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
from contracts.policies import PolicyIOSpec

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--socket", type=Path, required=True)
parser.add_argument("--contract", type=Path, required=True)
parser.add_argument("--task", default="Lab-VisualReach-FR3-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--idle-timeout", type=float, default=60)
parser.add_argument("--max-requests", type=int, default=1000)
parser.add_argument("--test-episode-length-s", type=float, help="Short timeout override for automatic-reset verification only")
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
import isaac_tasks.tasks  # noqa: E402,F401
_local_app_launcher = None


def main():
    global _local_app_launcher
    if not 1 <= args.max_requests <= 10000 or not 1 <= args.idle_timeout <= 300 or args.seed < 0:
        raise ValueError("Invalid request budget/idle timeout/seed")
    spec = PolicyIOSpec.from_json(args.contract.read_text())
    cfg, _ = resolve_task_config(args.task, "")
    cfg.scene.num_envs, cfg.seed = 1, args.seed
    if args.test_episode_length_s is not None:
        if not .15 <= args.test_episode_length_s <= 10:
            raise ValueError("Test episode duration must be .15..10 seconds")
        cfg.episode_length_s = args.test_episode_length_s
    if args.device is not None:
        cfg.sim.device = args.device
    from isaaclab.app import AppLauncher
    args.enable_cameras = True
    _local_app_launcher = AppLauncher(args)
    cfg.sim.device = _local_app_launcher.device
    with launch_simulation(cfg, args):
        from isaac_tasks.bridges.local_session import LocalVisualReachSession
        from isaac_tasks.bridges.local_server import serve
        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            session = LocalVisualReachSession(env, spec, args.seed)
            serve(session, args.socket, args.idle_timeout, args.max_requests)
            print("V2 simulation session closed cleanly", flush=True)
            return 0
        finally:
            env.close()
    return 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        if _local_app_launcher is not None:
            _local_app_launcher.app.close(exit_code=exit_code)
    raise SystemExit(exit_code)
