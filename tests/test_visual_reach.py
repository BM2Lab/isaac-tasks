"""Offline V1 semantics tests; simulator validation lives in the expert runner."""
import pytest
import torch
from isaac_tasks.controllers.visual_reach import expert_increment
from isaac_tasks.assets.fr3.visual_reach import MODEL, make_robot_cfg
from isaac_tasks.tasks.manager_based.reach.mdp.visual_reach import joint_positions


def test_expert_physical_norm_and_direction():
    p = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    goal = torch.tensor([[1., 1., 1.], [0., 0., 0.], [.001, 0., 0.]])
    delta = expert_increment(p, goal)
    assert torch.all(delta.norm(dim=-1) <= .00500001)
    assert torch.allclose(delta[1], torch.zeros(3))
    assert delta[2, 0] == pytest.approx(.0003)
    assert torch.allclose(delta[0]/delta[0].norm(), goal[0]/goal[0].norm())


def test_asset_requires_explicit_selection(monkeypatch):
    monkeypatch.delenv("ISAAC_TASKS_FR3_USD", raising=False)
    with pytest.raises(RuntimeError, match="ISAAC_TASKS_FR3_USD"):
        make_robot_cfg()


def test_asset_rejects_unverified_file(monkeypatch, tmp_path):
    usd = tmp_path/"wrong.usda"
    usd.write_text("not the audited model")
    monkeypatch.setenv("ISAAC_TASKS_FR3_USD", str(usd))
    with pytest.raises(ValueError, match="checksum"):
        make_robot_cfg()


def test_home_within_command_range():
    assert len(MODEL["joint_names"]) == 7
    assert all(lo < q < hi for lo, q, hi in zip(MODEL["command_lower_rad"], MODEL["home_rad"], MODEL["command_upper_rad"]))


def test_task_registration_does_not_load_asset(monkeypatch):
    import gymnasium as gym
    monkeypatch.delenv("ISAAC_TASKS_FR3_USD", raising=False)
    assert gym.spec("Lab-VisualReach-FR3-v0").kwargs["env_cfg_entry_point"].endswith(":Fr3VisualReachEnvCfg")


def test_measured_wrist_mount_is_synced_before_render():
    from types import SimpleNamespace as S
    calls = []
    position = torch.tensor([[[1., 2., 3.]], [[4., 5., 6.]]])
    rotation = torch.tensor([[[0., 0., 0., 1.]], [[0., 0., 0., 1.]]])
    robot = S(data=S(body_link_pos_w=S(torch=position), body_link_quat_w=S(torch=rotation),
        joint_pos=S(torch=torch.zeros(2, 7))))
    def set_pose(**kw):
        calls.append("pose")
        assert torch.allclose(kw["positions"], position[:, 0]+torch.tensor([.14, 0., -.03]))
        assert kw["convention"] == "ros"
    camera = S(cfg=S(mount_pos=(.14, 0., -.03), mount_rot_ros_xyzw=(0., 0., 0., 1.)), set_world_poses=set_pose)
    env = S(scene={"robot": robot, "wrist": camera}, device="cpu", num_envs=2,
        _visual_reach_tcp_body_id=0, sim=S(render=lambda: calls.append("render")))
    assert joint_positions(env).shape == (2, 7)
    assert calls == ["pose", "render"]
