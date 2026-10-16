"""Manager action adapter for P1: native world OSC and integrated gripper.

No camera/policy features yet. All simulator data enter the qualified float64
algebra explicitly; commands return to the simulator's float32 interface.
"""
import json
from pathlib import Path

import torch
import warp as wp
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import matrix_from_quat
from .libero_osc import LiberoOscController, OscState


class LiberoOscAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        manifest = json.loads(Path(cfg.scene_manifest).read_text())
        self.ref = manifest["reference"]
        self.arm_ids, names = self._asset.find_joints([f"robot0_joint{i}" for i in range(1, 8)], preserve_order=True)
        self.finger_ids, _ = self._asset.find_joints(["gripper0_finger_joint1", "gripper0_finger_joint2"], preserve_order=True)
        if len(names) != 7 or not self._asset.is_fixed_base:
            raise ValueError("P1 requires a fixed native Panda with seven arm joints")
        self.body = self._asset.find_bodies("robot0_link7")[0][0]
        self.dofs = [i+self._asset.num_base_dofs for i in self.arm_ids]
        self._raw = torch.zeros((self.num_envs, 7), device=self.device)
        self.gripper = torch.zeros((self.num_envs, 2), device=self.device)
        self.needs_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.anchor = torch.tensor(self.ref["model"]["initial_joint"], device=self.device, dtype=torch.float64).repeat(self.num_envs, 1)
        indices = self.ref["model"]["arm_qvel_indices"]
        self.armature = torch.tensor([self.ref["scene"]["dof_armature"][i] for i in indices], device=self.device, dtype=torch.float64)
        self.passive_damping = torch.tensor([self.ref["scene"]["dof_damping"][i] for i in indices], device=self.device)
        self.passive_friction = torch.tensor([self.ref["scene"]["dof_frictionloss"][i] for i in indices], device=self.device)
        # Obtain the fixed TCP transform from native body poses, not a Panda
        # hand/grip-site convention guessed from a different USD.
        poses = self.ref["scene"]["body_pose_wxyz"]
        def pose(name):
            p = torch.tensor(poses[name], device=self.device, dtype=torch.float64)
            return p[:3], matrix_from_quat(p[[4, 5, 6, 3]])
        link_p, link_r = pose("robot0_link7")
        tip_p, tip_r = pose("gripper0_eef")
        self.offset = link_r.T @ (tip_p-link_p)
        self.tip_rotation = link_r.T @ tip_r
        self.controller = LiberoOscController(self.state(), self.anchor,
            self.ref["torque_lower"], self.ref["torque_upper"])
        self.last_raw_torque = torch.zeros((self.num_envs, 7), dtype=torch.float64, device=self.device)

    @property
    def action_dim(self):
        return 7

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._raw.clamp(-1, 1)

    def state(self):
        data = self._asset.data
        rot = matrix_from_quat(data.body_link_quat_w.torch[:, self.body]).double()
        offset = (rot @ self.offset.expand(self.num_envs, 3).unsqueeze(-1)).squeeze(-1)
        pos = data.body_link_pos_w.torch[:, self.body].double()+offset-self._env.scene.env_origins.double()
        j = data.body_link_jacobian_w.torch[:, self.body-1, :, self.dofs].double().clone()
        j[:, :3] += torch.linalg.cross(j[:, 3:].transpose(1, 2), offset[:, None, :], dim=-1).transpose(1, 2)
        dq = data.joint_vel.torch[:, self.arm_ids].double()
        twist = (j @ dq.unsqueeze(-1)).squeeze(-1)
        raw_c = self._asset.root_view.get_coriolis_and_centrifugal_compensation_forces()
        c = wp.to_torch(raw_c) if isinstance(raw_c, wp.array) else raw_c
        bias = (data.gravity_compensation_forces.torch[:, self.dofs]+c[:, self.dofs]).double()
        return OscState(pos, rot @ self.tip_rotation, twist[:, :3], twist[:, 3:],
            data.joint_pos.torch[:, self.arm_ids].double(), dq,
            # PhysX's queried generalized mass omits the rotor armature even
            # though the solver uses it; MuJoCo qM includes it.
            data.mass_matrix.torch[:, self.dofs][:, :, self.dofs].double()+torch.diag_embed(self.armature), j, bias)

    def process_actions(self, actions):
        if actions.shape != self._raw.shape or not torch.isfinite(actions).all():
            raise ValueError("Expected finite Nx7 native controller actions")
        self._raw.copy_(actions)
        state = self.state()
        # Fixed-size masking avoids a GPU nonzero()/dynamic-index sync at each
        # policy boundary; neighboring controller state is left untouched.
        mask = self.needs_reset[:, None]
        self.controller.goal_position.copy_(torch.where(mask, state.position, self.controller.goal_position))
        self.controller.goal_rotation.copy_(torch.where(mask[:, :, None], state.rotation, self.controller.goal_rotation))
        self.controller.initial_joint.copy_(torch.where(mask, self.anchor, self.controller.initial_joint))
        self.needs_reset[:] = False
        self.controller.set_goal(actions[:, :6].double(), state)
        self.gripper.add_(self.gripper.new_tensor([-1., 1.]) * .01 * actions[:, 6:7].sign()).clamp_(-1, 1)

    def apply_actions(self):
        result = self.controller.compute(self.state())
        self.last_raw_torque.copy_(result["raw_torque"])
        dq_arm = self._asset.data.joint_vel.torch[:, self.arm_ids]
        # Kinetic friction approximation, not MuJoCo's solver-level static
        # friction constraint. Avoid treating frictionloss [Nm] as a unitless
        # PhysX joint-load friction coefficient.
        passive = self.passive_damping*dq_arm + self.passive_friction*dq_arm.sign()
        self._asset.set_joint_effort_target_index(target=result["torque"].float()-passive, joint_ids=self.arm_ids)
        # Native sign-integrated gripper setpoints; arm and finger controls do
        # not share a torque scale. Native motor limit is 20 N per finger.
        target = (self.gripper+1)*.02 + self.gripper.new_tensor([0., -.04])
        q = self._asset.data.joint_pos.torch[:, self.finger_ids]
        dq = self._asset.data.joint_vel.torch[:, self.finger_ids]
        effort = (1000*(target-q)).clamp(-20, 20)-100*dq-dq.sign()
        self._asset.set_joint_effort_target_index(target=effort, joint_ids=self.finger_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.gripper[ids] = 0
        self.needs_reset[ids] = True


@configclass
class LiberoOscActionCfg(ActionTermCfg):
    class_type: type = LiberoOscAction
    asset_name: str = "robot"
    scene_manifest: str = ""
