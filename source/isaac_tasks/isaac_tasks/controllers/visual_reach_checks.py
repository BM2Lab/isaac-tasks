"""Bounded initialization diagnostics, never called in the control hot path."""
import torch
from isaac_tasks.controllers.visual_reach import tcp_pose_base


def verify_fk_and_clone_parity(env):
    robot = env.scene["robot"]
    action = env.action_manager.get_term("cartesian_increment")
    q0 = robot.data.joint_pos.torch.clone()
    jacobian = robot.data.body_link_jacobian_w.torch[:, action.body_id-1, :3, action.dof_ids].clone()
    numerical = torch.zeros_like(jacobian)
    epsilon = 0.001
    # Seven diagnostic perturbations; no physics steps, forces or image captures.
    for column, joint_id in enumerate(action.joint_ids):
        positions = []
        for sign in (-1, 1):
            q = q0.clone()
            q[:, joint_id] += sign*epsilon
            robot.write_joint_position_to_sim_index(position=q)
            env.sim.forward()
            robot.update(env.physics_dt)
            positions.append(robot.data.body_link_pos_w.torch[:, action.body_id].clone())
        numerical[:, :, column] = (positions[1]-positions[0])/(2*epsilon)
    fk_difference = float((numerical-jacobian).abs().max())
    env.reset()
    trace = []
    delta = torch.zeros((env.num_envs, 3), device=env.device)
    delta[:, 2] = 0.001
    clone_difference = 0.0
    for _ in range(5):
        env.step(delta)
        q = robot.data.joint_pos.torch.clone()
        clone_difference = max(clone_difference, float((q-q[:1]).abs().max()))
        trace.append(q[0].cpu().tolist())
    env.reset()
    return {"fk_jacobian_max_difference_m_rad": fk_difference,
        "clone_max_joint_difference_rad": clone_difference, "single_clone_trace_rad": trace}, {
        "link_tcp_jacobian_finite_difference": fk_difference < 0.001,
        "clone_parity": clone_difference < 1e-5}
