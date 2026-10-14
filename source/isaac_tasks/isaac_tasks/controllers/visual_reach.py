"""Physical Cartesian increments and a privileged scripted V1 IK expert."""
import torch
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv
from isaac_tasks.assets.fr3.visual_reach import MODEL


def tcp_pose_base(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_visual_reach_tcp_body_id"):
        env._visual_reach_tcp_body_id = robot.find_bodies(MODEL["tcp_body"])[0][0]
    body_id = env._visual_reach_tcp_body_id
    return subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
        robot.data.body_link_pos_w.torch[:, body_id], robot.data.body_link_quat_w.torch[:, body_id])


def expert_increment(position, target, max_norm_m=0.005, gain=0.3):
    """Output meters, not normalized policy coordinates."""
    delta = gain * (target - position)
    return delta * (max_norm_m / delta.norm(dim=-1, keepdim=True).clamp_min(max_norm_m))


class CartesianIncrementAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, names = self._asset.find_joints(MODEL["joint_names"], preserve_order=True)
        if names != MODEL["joint_names"] or self._asset.num_joints != 7 or not self._asset.is_fixed_base:
            raise ValueError("V1 requires the fixed-base seven-joint bare-flange FR3")
        self.body_id = self._asset.find_bodies(MODEL["tcp_body"])[0][0]
        self.dof_ids = [j + self._asset.num_base_dofs for j in self.joint_ids]
        self._raw = torch.zeros((self.num_envs, 3), device=self.device)
        self._processed = torch.zeros_like(self._raw)
        self.anchor = torch.zeros((self.num_envs, 4), device=self.device)
        self.needs_anchor = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.target = torch.zeros((self.num_envs, 7), device=self.device)
        self.lower = torch.tensor(MODEL["command_lower_rad"], device=self.device)
        self.upper = torch.tensor(MODEL["command_upper_rad"], device=self.device)
        self.projection = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"), self.num_envs, self.device)

    @property
    def action_dim(self):
        return 3

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def process_actions(self, actions):
        # Policy-boundary validation; never silently normalize incompatible actions.
        if actions.shape != self._raw.shape or not torch.isfinite(actions).all() or (actions.norm(dim=-1) > self.cfg.max_norm_m + 1e-7).any():
            raise ValueError("Expected finite Nx3 Cartesian increments with L2 norm <= 0.005 m")
        pos, quat = tcp_pose_base(self._env)
        self.anchor[:] = torch.where(self.needs_anchor[:, None], quat, self.anchor)
        self.needs_anchor[:] = False
        self._raw[:] = actions
        self._processed[:] = actions
        self.target[:, :3] = pos + actions
        self.target[:, 3:] = self.anchor
        self.ik.set_command(self.target)

    def apply_actions(self):
        pos, quat = tcp_pose_base(self._env)
        jacobian = self._asset.data.body_link_jacobian_w.torch[:, self.body_id - 1, :, self.dof_ids].clone()
        rotation = matrix_from_quat(quat_inv(self._asset.data.root_quat_w.torch))
        jacobian[:, :3] = rotation @ jacobian[:, :3]
        jacobian[:, 3:] = rotation @ jacobian[:, 3:]
        q = self._asset.data.joint_pos.torch[:, self.joint_ids]
        requested = self.ik.compute(pos, quat, jacobian, q)
        desired = requested.clamp(self.lower, self.upper)
        self.projection[:] = (desired != requested).any(dim=-1)
        self._asset.set_joint_position_target_index(target=desired, joint_ids=self.joint_ids)
        # Gravity is physically enabled. Feedforward offsets static sag of implicit PD.
        self._asset.set_joint_effort_target_index(
            target=self._asset.data.gravity_compensation_forces.torch[:, self.dof_ids], joint_ids=self.joint_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self._processed[ids] = 0
        self.target[ids] = 0
        self.anchor[ids] = 0
        self.needs_anchor[ids] = True
        self.projection[ids] = False


@configclass
class CartesianIncrementActionCfg(ActionTermCfg):
    class_type: type = CartesianIncrementAction
    asset_name: str = "robot"
    max_norm_m: float = 0.005
