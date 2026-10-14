"""Vectorized visual-reaching manager terms. Target is privileged, not policy input."""
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, combine_frame_transforms
from isaac_tasks.controllers.visual_reach import tcp_pose_base

INSTRUCTION = "Move the end effector to the red marker."


class VisibleTargetCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._target = torch.zeros((self.num_envs, 3), device=self.device)
        self.low = torch.tensor(cfg.lower, device=self.device)
        self.high = torch.tensor(cfg.upper, device=self.device)
        self.metrics["position_error_m"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self._target

    def _resample_command(self, env_ids):
        self._target[env_ids] = self.low + torch.rand((len(env_ids), 3), device=self.device) * (self.high-self.low)
        robot = self._env.scene["robot"]
        pose = torch.zeros((len(env_ids), 7), device=self.device)
        pose[:, :3] = robot.data.root_pos_w.torch[env_ids] + quat_apply(
            robot.data.root_quat_w.torch[env_ids], self._target[env_ids])
        pose[:, 6] = 1
        self._env.scene["target_marker"].write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)

    def _update_metrics(self):
        self.metrics["position_error_m"][:] = position_error(self._env)

    def _update_command(self):
        pass


@configclass
class VisibleTargetCommandCfg(CommandTermCfg):
    class_type: type = VisibleTargetCommand
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    lower: tuple[float, float, float] = (0.35, -0.08, 0.40)
    upper: tuple[float, float, float] = (0.46, 0.08, 0.53)


def joint_positions(env):
    # This term is first in the policy group: synchronize *before either RGB
    # term renders. RTX deduplicates rendering between cameras within a step.
    sync_wrist_camera(env)
    return env.scene["robot"].data.joint_pos.torch.clone()


def sync_wrist_camera(env):
    # Explicit measured mount avoids stale PhysX/Fabric child transforms.
    sensor, robot = env.scene["wrist"], env.scene["robot"]
    if not hasattr(env, "_wrist_mount"):
        env._wrist_mount = (
            torch.tensor(sensor.cfg.mount_pos, device=env.device).expand(env.num_envs, -1),
            torch.tensor(sensor.cfg.mount_rot_ros_xyzw, device=env.device).expand(env.num_envs, -1))
    if not hasattr(env, "_visual_reach_tcp_body_id"):
        env._visual_reach_tcp_body_id = robot.find_bodies("fr3_flange")[0][0]
    body_id = env._visual_reach_tcp_body_id
    pos, quat = combine_frame_transforms(robot.data.body_link_pos_w.torch[:, body_id],
        robot.data.body_link_quat_w.torch[:, body_id], *env._wrist_mount)
    sensor.set_world_poses(positions=pos, orientations=quat, convention="ros")
    # Advance render generation after changing a camera pose. In visible Kit
    # mode this also pumps rendering; headless sensor rendering occurs on data
    # access. No physics step is taken here.
    env.sim.render()


def rgb(env, camera):
    return env.scene[camera].data.output["rgb"].torch[..., :3].clone()


def position_error(env):
    return (tcp_pose_base(env)[0] - env.command_manager.get_command("target")).norm(dim=-1)


def distance_reward(env):
    return torch.exp(-position_error(env)/0.1)


def reset_success(env, env_ids):
    if hasattr(env, "reach_streak"):
        env.reach_streak[env_ids] = 0


def sustained_success(env, threshold=0.02, steps=5):
    if not hasattr(env, "reach_streak"):
        env.reach_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.reach_terminal_error = position_error(env)
    env.reach_streak[:] = torch.where(env.reach_terminal_error < threshold, env.reach_streak+1, 0)
    return env.reach_streak >= steps


def invalid_state(env):
    robot = env.scene["robot"]
    q, v = robot.data.joint_pos.torch, robot.data.joint_vel.torch
    bounds = robot.data.soft_joint_pos_limits.torch
    return (~torch.isfinite(q).all(dim=-1) | ~torch.isfinite(v).all(dim=-1)
        | (q < bounds[..., 0]-0.01).any(dim=-1) | (q > bounds[..., 1]+0.01).any(dim=-1))
