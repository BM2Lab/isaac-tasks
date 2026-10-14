"""Device-resident V1 feature mapping. IPC and LeRobot preprocessing are deferred to V2."""
import torch
from contracts.policies import PolicyIOSpec
from isaac_tasks.assets.fr3.visual_reach import MODEL
from isaac_tasks.tasks.manager_based.reach.mdp.visual_reach import INSTRUCTION
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_apply, quat_apply_inverse


class VisualReachBridge:
    def __init__(self, env, spec: PolicyIOSpec):
        self.env, self.spec = env, spec
        if (spec.contract_id != "fr3_visual_reach_v1" or spec.robot.model != MODEL["model"]
            or list(spec.robot.joint_names) != MODEL["joint_names"]
            or list(env.scene["robot"].joint_names) != MODEL["joint_names"]
            or spec.robot.base_frame != "robot_base" or spec.robot.tcp_frame != "bare_flange"
            or spec.action.policy_hz != 1/env.step_dt or spec.action.max_translation_norm_m != 0.005):
            raise ValueError("Simulator and V1 physical policy contract do not match")
        self.cameras = {"observation.images.overhead": "overhead", "observation.images.wrist": "wrist"}
        if {c.key for c in spec.observation.cameras} != set(self.cameras):
            raise ValueError("V1 requires exactly overhead and wrist RGB cameras")
        for camera in spec.observation.cameras:
            if camera.optical_frame != self.cameras[camera.key] + "_optical":
                raise ValueError("Camera optical-frame mapping differs from V1")
            cfg = env.scene[self.cameras[camera.key]].cfg
            if (camera.height, camera.width) != (cfg.height, cfg.width):
                raise ValueError("Camera geometry differs from policy contract")

    def policy_features(self, observations):
        policy = observations["policy"]
        q = policy["joint_positions"]
        if q.shape != (self.env.num_envs, 7) or not torch.isfinite(q).all():
            raise ValueError("Invalid joint observation")
        result = {self.spec.observation.state_key: q, self.spec.observation.instruction_key: INSTRUCTION}
        for camera in self.spec.observation.cameras:
            rgb = policy[self.cameras[camera.key]]
            if rgb.shape != (self.env.num_envs, camera.height, camera.width, 3) or rgb.dtype != torch.uint8:
                raise ValueError("Camera must provide uint8 NHWC RGB")
            result[camera.key] = rgb
        return result

    def frame_counters(self):
        return torch.stack([self.env.scene[name].frame.torch.clone() for name in self.cameras.values()], dim=-1)

    def camera_metadata(self):
        """Actual capture poses/intrinsics; diagnostic metadata, never policy input."""
        return {key: {"position_world_m": self.env.scene[name].data.pos_w.torch.clone(),
            "quaternion_world_ros_xyzw": self.env.scene[name].data.quat_w_ros.torch.clone(),
            "intrinsics": self.env.scene[name].data.intrinsic_matrices.torch.clone(),
            "frame": self.env.scene[name].frame.torch.clone()}
            for key, name in self.cameras.items()}

    def wrist_mount_error(self):
        robot, camera = self.env.scene["robot"], self.env.scene["wrist"]
        body_id = self.env._visual_reach_tcp_body_id
        pos, quat = combine_frame_transforms(robot.data.body_link_pos_w.torch[:, body_id],
            robot.data.body_link_quat_w.torch[:, body_id], *self.env._wrist_mount)
        return (pos-camera.data.pos_w.torch).norm(dim=-1), quat_error_magnitude(quat, camera.data.quat_w_ros.torch)

    def marker_pixel_error(self, features, camera_key):
        """Privileged initialization check that RGB agrees with capture extrinsics."""
        robot = self.env.scene["robot"]
        target_world = robot.data.root_pos_w.torch + quat_apply(robot.data.root_quat_w.torch,
            self.env.command_manager.get_command("target"))
        sensor = self.env.scene[self.cameras[camera_key]]
        data = sensor.data
        point = quat_apply_inverse(data.quat_w_ros.torch, target_world-data.pos_w.torch)
        uvw = torch.bmm(data.intrinsic_matrices.torch, point.unsqueeze(-1)).squeeze(-1)
        uv = uvw[:, :2]/uvw[:, 2:].clamp_min(1e-6)
        image = features[camera_key].float()
        red = (image[..., 0] > 80) & (image[..., 0] > 1.5*image[..., 1]) & (image[..., 0] > 1.5*image[..., 2])
        y, x = torch.meshgrid(torch.arange(image.shape[1], device=self.env.device),
            torch.arange(image.shape[2], device=self.env.device), indexing="ij")
        distance = ((x-uv[:, 0, None, None])**2+(y-uv[:, 1, None, None])**2).sqrt()
        error = distance.masked_fill(~red, float("inf")).flatten(1).min(dim=-1).values
        return error.masked_fill(point[:, 2] <= 0, float("inf"))
