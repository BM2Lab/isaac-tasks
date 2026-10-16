"""P2 native EE/gripper state and processed-image mapping; no ML imports."""
import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix


def quaternion_rotation_vector(quat):
    """LeRobot float32 formula; deliberately do NOT canonicalize quaternion sign."""
    quat = quat.float()
    w = quat[:, 3].clamp(-1, 1)
    denominator = (1-w*w).clamp_min(0).sqrt()
    scale = torch.where(denominator > 1e-10, 2*w.acos()/denominator.clamp_min(1e-10), 0.)
    return quat[:, :3]*scale[:, None]


def rotation_vector(rotation):
    """Canonical matrix utility; policy features instead use native FK below."""
    quat = quat_from_matrix(rotation)
    return quaternion_rotation_vector(torch.where(quat[:, 3:4] < 0, -quat, quat))


def native_orientation(joint_position, chain):
    """Compose native body/joint quaternions without losing the sign branch."""
    def multiply(a, b):
        return torch.cat((a[:, 3:]*b[:, :3]+b[:, 3:]*a[:, :3]+torch.linalg.cross(a[:, :3], b[:, :3]),
                          a[:, 3:]*b[:, 3:]-(a[:, :3]*b[:, :3]).sum(-1, keepdim=True)), -1)
    result = joint_position.new_tensor([0., 0., 0., 1.]).expand(len(joint_position), 4)
    for fixed, index, axis in chain:  # fixed robot links, never environment loops
        result = multiply(result, fixed.expand_as(result))
        if index is not None:
            half_angle = joint_position[:, index:index+1]/2
            variable = torch.cat((axis*half_angle.sin(), half_angle.cos()), -1)
            result = multiply(result, variable)
    return result


def synchronize_cameras(env):
    robot = env.scene["robot"]
    action = env.action_manager.get_term("osc")
    link_r = matrix_from_quat(robot.data.body_link_quat_w.torch[:, action.body])
    link_p = robot.data.body_link_pos_w.torch[:, action.body]
    for name in ("agentview", "eye_in_hand"):
        sensor = env.scene[name]
        cfg = sensor.cfg
        if cfg.mounted:
            offset = link_p.new_tensor(cfg.mount_position)
            rotation = link_p.new_tensor(cfg.mount_rotation).expand(env.num_envs, 3, 3)
            pos = link_p+(link_r@offset.expand(env.num_envs, 3)[..., None]).squeeze(-1)
            quat = quat_from_matrix(link_r@rotation)
        else:
            pos = link_p.new_tensor(cfg.reference_position)+env.scene.env_origins
            quat = quat_from_matrix(link_p.new_tensor(cfg.reference_rotation)).expand(env.num_envs, 4)
        sensor.set_world_poses(positions=pos, orientations=quat, convention="opengl")
    # Render generation after measured camera poses, before RGB data access.
    # Rendering does not advance physics; repeated observe is served from cache.
    env.sim.render()


def state_observation(env):
    synchronize_cameras(env)
    action = env.action_manager.get_term("osc")
    state = action.state()
    if not hasattr(env, "_libero_orientation_chain"):
        env._libero_orientation_chain = [(state.position.new_tensor(item["fixed_quat_xyzw"]),
            item["joint_index"], state.position.new_tensor(item["axis"]))
            for item in env.cfg.libero_orientation_chain]
    orientation = native_orientation(state.joint_position, env._libero_orientation_chain)
    return torch.cat((state.position.float(), quaternion_rotation_vector(orientation),
                      env.scene["robot"].data.joint_pos.torch[:, action.finger_ids]), -1)


def rgb_observation(env, camera):
    rgb = env.scene[camera].data.output["rgb"].torch[..., :3]
    # Native readPixels is bottom-up; LeRobot then flips H AND W. Isaac is
    # already top-down. Only W remains: a second H flip would invert the view.
    return rgb.flip(dims=(2,)).contiguous()
