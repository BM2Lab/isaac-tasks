"""Batched robosuite 1.4 OSC_POSE algebra; no simulator or hardware imports.

Native world frame, Jacobian rows [linear, angular], rotation matrices (not
quaternions), arm joints only. Bias must include gravity AND Coriolis. This is
the fixed-gain, delta-goal, no-interpolation profile, not a general OSC adapter.
See docs/p0_libero_osc.md for provenance and numerical limitations.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class OscState:
    position: torch.Tensor
    rotation: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    mass: torch.Tensor
    jacobian: torch.Tensor
    bias: torch.Tensor


def orientation_error(desired, current):
    """Native cross-column error (sin(theta)*axis), NOT an SO(3) logarithm."""
    return 0.5 * torch.linalg.cross(current, desired, dim=-2).sum(dim=-1)


def _native_delta_rotation(vector):
    # robosuite axisangle2quat uses float64, quat2mat casts to float32 before
    # its normalized outer product. Preserve that otherwise surprising detail.
    angle = vector.norm(dim=-1, keepdim=True)
    axis = vector / angle.clamp_min(torch.finfo(vector.dtype).tiny)
    quat = torch.cat((torch.cos(angle / 2), axis * torch.sin(angle / 2)), -1).float()
    quat = quat * torch.sqrt(2.0 / (quat * quat).sum(-1, keepdim=True))
    outer = quat.unsqueeze(-1) * quat.unsqueeze(-2)
    w, x, y, z = 0, 1, 2, 3
    return torch.stack((
        1-outer[..., y, y]-outer[..., z, z], outer[..., x, y]-outer[..., z, w], outer[..., x, z]+outer[..., y, w],
        outer[..., x, y]+outer[..., z, w], 1-outer[..., x, x]-outer[..., z, z], outer[..., y, z]-outer[..., x, w],
        outer[..., x, z]-outer[..., y, w], outer[..., y, z]+outer[..., x, w], 1-outer[..., x, x]-outer[..., y, y],
    ), -1).reshape(*vector.shape[:-1], 3, 3).to(vector.dtype)


class LiberoOscController:
    """Held Cartesian goals at policy rate; compute torque at physics rate.

    Construct/reset outside the hot path. Caller validates finite inputs, SPD M,
    joint order and shapes. No hidden filtering, slew limiter or IK. Native
    robot torque clipping is exposed separately from the raw controller torque.
    """

    def __init__(self, initial: OscState, initial_joint, torque_lower, torque_upper,
                 *, uncoupling=True, kp=150.0, pinv_rtol=1e-15):
        if initial.position.dtype != torch.float64:
            raise ValueError("P0 reference profile requires float64; float32 SVD cutoff is not qualified")
        self.goal_position = initial.position.clone()
        self.goal_rotation = initial.rotation.clone()
        self.initial_joint = initial_joint.clone()
        self.lower = torch.as_tensor(torque_lower, device=initial.position.device, dtype=initial.position.dtype)
        self.upper = torch.as_tensor(torque_upper, device=initial.position.device, dtype=initial.position.dtype)
        self.kp = kp
        self.kd = 2 * math.sqrt(kp)
        self.uncoupling = uncoupling
        self.pinv_rtol = pinv_rtol
        self.scale = initial.position.new_tensor([.05, .05, .05, .5, .5, .5])
        self.eye = torch.eye(initial.joint_position.shape[-1], device=initial.position.device,
                             dtype=initial.position.dtype)

    def reset(self, state: OscState, initial_joint, env_ids):
        """Selective reset; preserve neighbors' goals and null-space anchors."""
        self.goal_position[env_ids] = state.position[env_ids]
        self.goal_rotation[env_ids] = state.rotation[env_ids]
        self.initial_joint[env_ids] = initial_joint[env_ids]

    def set_goal(self, action, state: OscState):
        delta = action.clamp(-1, 1) * self.scale
        self.goal_position.copy_(state.position + delta[..., :3])
        proposed = _native_delta_rotation(delta[..., 3:]) @ state.rotation
        changed = (delta[..., 3:] != 0).any(dim=-1)
        self.goal_rotation.copy_(torch.where(changed[:, None, None], proposed, self.goal_rotation))

    def compute(self, state: OscState):
        j = state.jacobian
        # Native np.linalg.pinv default cutoff, explicitly not Torch's default.
        minv_jt = torch.linalg.solve(state.mass, j.transpose(-1, -2))
        inv_lambda = j @ minv_jt
        full = torch.linalg.pinv(inv_lambda, atol=0.0, rtol=self.pinv_rtol)
        desired = torch.cat((
            self.kp * (self.goal_position-state.position) - self.kd*state.linear_velocity,
            self.kp * orientation_error(self.goal_rotation, state.rotation) - self.kd*state.angular_velocity,
        ), -1)
        if self.uncoupling:
            pos = torch.linalg.pinv(inv_lambda[..., :3, :3], atol=0.0, rtol=self.pinv_rtol)
            rot = torch.linalg.pinv(inv_lambda[..., 3:, 3:], atol=0.0, rtol=self.pinv_rtol)
            wrench = torch.cat(((pos @ desired[..., :3, None]).squeeze(-1),
                                (rot @ desired[..., 3:, None]).squeeze(-1)), -1)
        else:
            wrench = (full @ desired.unsqueeze(-1)).squeeze(-1)
        null = self.eye - minv_jt @ full @ j
        posture = 10*(self.initial_joint-state.joint_position) - 2*math.sqrt(10)*state.joint_velocity
        raw = (j.transpose(-1, -2) @ wrench.unsqueeze(-1)
               + null.transpose(-1, -2) @ state.mass @ posture.unsqueeze(-1)).squeeze(-1) + state.bias
        return {"raw_torque": raw, "torque": torch.maximum(torch.minimum(raw, self.upper), self.lower),
                "nullspace": null}
