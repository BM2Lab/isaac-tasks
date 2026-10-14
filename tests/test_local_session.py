"""Automatic-reset identity tests with a mock environment, no Kit launch."""
from types import SimpleNamespace as S
import torch
from contracts.actions import ActionChunk, CartesianDeltaSpec
from contracts.observations import CameraSpec, ObservationSpec
from contracts.policies import PolicyIOSpec
from contracts.robot import RobotSpec
from contracts.samples import ObservationPacket
from isaac_tasks.bridges import local_session


def test_auto_reset_closes_old_episode_and_discards_pending_rows(monkeypatch):
    joints = tuple(f"panda_joint{i}" for i in range(1, 8))
    spec = PolicyIOSpec("contract", RobotSpec("sim", "fr3", "arm", joints, "base", "tip"),
        ObservationSpec("state", joints, (CameraSpec("camera", "optical", 1, 1),), "task"),
        CartesianDeltaSpec("arm", "base", "tip", 20, .005), 50)
    class Bridge:
        cameras = {"camera": "camera"}
        def __init__(self, env, spec):
            pass
        def policy_features(self, observations):
            return {"state": torch.zeros(1, 7), "camera": torch.zeros(1, 1, 1, 3, dtype=torch.uint8), "task": "move"}
    monkeypatch.setattr(local_session, "VisualReachBridge", Bridge)
    env = S(num_envs=1, step_dt=.05, device="cpu", reset=lambda seed=None: ({}, {}),
        scene={"camera": S(frame=S(torch=torch.tensor([0])))},
        step=lambda action: ({}, torch.zeros(1), torch.tensor([False]), torch.tensor([True]), {}),
        reach_terminal_error=torch.tensor([.1]), termination_manager=S(get_term=lambda name: torch.tensor([False])))
    session = local_session.LocalVisualReachSession(env, spec)
    source = session.context
    session.submit(ActionChunk(source, spec.action, ((0., 0., 0.),)*3, 1_000_000_000).to_dict())
    result = session.advance(source.to_dict())
    packet = ObservationPacket.from_dict(result["observation"])
    assert result["terminal"]["episode_id"] == source.episode_id
    assert result["terminal"]["timeout"]
    assert packet.context.episode_id != source.episode_id
    assert packet.context.step_index == 0
    assert packet.context.timestamp_ns == 50_000_000
    assert result["status"]["queued_actions"] == 0


def test_expert_label_rpc_is_disabled_by_default():
    import pytest
    session = object.__new__(local_session.LocalVisualReachSession)
    session.enable_expert = False
    with pytest.raises(ValueError, match="disabled"):
        session.expert_action({})
