"""Batched algebra and state lifecycle; independent of simulator launch."""
from pathlib import Path
from hashlib import sha256
import json

import pytest
import torch

from isaac_tasks.controllers.libero_osc import LiberoOscController, OscState, orientation_error
from isaac_tasks.controllers.libero_osc_checks import check_reference


def state(n=3):
    z = torch.zeros(n, 3, dtype=torch.float64)
    return OscState(z.clone(), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1), z.clone(), z.clone(),
                    torch.zeros(n, 7, dtype=torch.float64), torch.zeros(n, 7, dtype=torch.float64),
                    torch.eye(7, dtype=torch.float64).repeat(n, 1, 1),
                    torch.eye(6, 7, dtype=torch.float64).repeat(n, 1, 1), torch.ones(n, 7, dtype=torch.float64))


def make(s):
    return LiberoOscController(s, s.joint_position, [-87]*7, [87]*7)


def test_native_reference():
    result = check_reference(Path(__file__).parent / "fixtures/libero_osc_reference.json")
    assert result["passed"], result


def test_bias_posture_and_clipping():
    s = state()
    c = make(s)
    assert torch.equal(c.compute(s)["torque"], s.bias)
    s.joint_position[:, 6] = .2
    assert c.compute(s)["raw_torque"][0, 6] == pytest.approx(-1)
    s.bias[:, 0] = 200
    assert c.compute(s)["torque"][0, 0] == 87


def test_zero_rotation_latches_and_position_reanchors():
    s = state()
    c = make(s)
    a = torch.zeros(3, 6, dtype=torch.float64)
    a[:, 5] = .4
    c.set_goal(a, s)
    old = c.goal_rotation.clone()
    s.position += .2
    s.rotation[:] = torch.eye(3, dtype=torch.float64)
    c.set_goal(a*0, s)
    assert torch.equal(c.goal_rotation, old)
    assert torch.equal(c.goal_position, s.position)


def test_selective_reset_and_batch_single_parity():
    s = state()
    c = make(s)
    a = torch.arange(18, dtype=torch.float64).reshape(3, 6)/10
    c.set_goal(a, s)
    before = c.compute(s)["torque"].clone()
    goals = c.goal_position.clone()
    s.position[1] += .3
    s.joint_position[1] += .1
    c.reset(s, s.joint_position, torch.tensor([1]))
    assert torch.equal(c.goal_position[[0, 2]], goals[[0, 2]])
    assert torch.equal(c.initial_joint[1], s.joint_position[1])
    assert torch.equal(c.compute(s)["torque"][[0, 2]], before[[0, 2]])
    single = state(1)
    one = make(single)
    one.set_goal(a[:1], single)
    assert torch.allclose(one.compute(single)["torque"][0], before[0])


def test_native_pi_error_is_zero_not_log_map():
    current = torch.eye(3, dtype=torch.float64)[None]
    desired = torch.diag(torch.tensor([1., -1., -1.], dtype=torch.float64))[None]
    assert torch.equal(orientation_error(desired, current), torch.zeros(1, 3, dtype=torch.float64))


def test_unqualified_precision_rejected():
    s = state()
    s.position = s.position.float()
    with pytest.raises(ValueError, match="float64"):
        make(s)


def test_full_export_receipt_rejects_tampering(tmp_path):
    fixture = Path(__file__).parent / "fixtures/libero_osc_reference.json"
    data = json.loads(fixture.read_text())
    data["model"] = {"xml_sha256": sha256(b"model").hexdigest()}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(data))
    (tmp_path / "receipt.json").write_text(json.dumps({"reference_sha256": sha256(path.read_bytes()).hexdigest()}))
    (tmp_path / "model.xml").write_bytes(b"model")
    assert check_reference(path)["passed"]
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="Reference receipt"):
        check_reference(path)
