import math
import json
from pathlib import Path
from types import SimpleNamespace as S

import pytest
import torch

from contracts.libero import LiberoIOSpec, LiberoActionChunk
from contracts.episodes import SampleContext
from isaac_tasks.bridges.action_queue import ActionQueue
from isaac_tasks.bridges.libero_visual import rgb_observation, rotation_vector
from isaac_tasks.bridges.libero_visual import native_orientation, quaternion_rotation_vector
from isaac_tasks.bridges.libero_session import LocalLiberoSession


def test_faulted_session_requires_restart():
    session = object.__new__(LocalLiberoSession)
    session.failed = True
    with pytest.raises(RuntimeError, match="faulted"):
        session.reset()


@pytest.mark.parametrize("horizon", [True, 1.5, 0, 281])
def test_session_rejects_bad_horizons_before_scene_access(horizon):
    with pytest.raises(ValueError):
        LocalLiberoSession(S(num_envs=1, step_dt=.05), "unused", max_steps=horizon)


def test_native_observation_body_and_quaternion_branch():
    ref = json.loads((Path(__file__).parent/"fixtures/libero_visual_reference.json").read_text())
    q = torch.tensor([ref["joint_position"]], dtype=torch.float64)
    chain = [(q.new_tensor(item["fixed_quat_xyzw"]), item["joint_index"], q.new_tensor(item["axis"]))
             for item in ref["orientation_chain"]]
    actual = quaternion_rotation_vector(native_orientation(q, chain))
    # The cached reset observation differs slightly in time from current q/dq.
    # Fresh same-state parity is checked by the simulator qualification CLI.
    assert torch.allclose(actual[0], torch.tensor(ref["state"][3:6]), atol=2e-4)
    # Sign is semantically meaningful to upstream, unlike SO(3) log maps.
    raw = torch.tensor([[0., 0., math.sin(3.5/2), math.cos(3.5/2)]])
    assert quaternion_rotation_vector(raw)[0, 2] == pytest.approx(3.5, abs=1e-6)


def test_isaac_top_down_to_native_processed_pixels():
    # Native raw H is inverted relative to Isaac. Official upstream then flips
    # H and W: net result is one W flip from Isaac, NOT a 180-degree rotation.
    pixels = torch.arange(18, dtype=torch.uint8).reshape(1, 2, 3, 3)
    native_raw = pixels.flip((1,))
    expected = native_raw.flip((1, 2))
    env = S(scene={"camera": S(data=S(output={"rgb": S(torch=pixels)}))})
    assert torch.equal(rgb_observation(env, "camera"), expected)


def test_rotation_vector_known_conventions():
    r = torch.tensor([[[1., 0, 0], [0, 1, 0], [0, 0, 1]],
                      [[0., -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=torch.float64)
    actual = rotation_vector(r)
    assert actual.dtype == torch.float32
    assert torch.equal(actual[0], torch.zeros(3))
    assert torch.allclose(actual[1], torch.tensor([0., 0., math.pi/2]), atol=1e-6)


def test_native_queue_execution_expiry_and_reset():
    spec = LiberoIOSpec()
    queue = ActionQueue(spec)
    context = SampleContext("e", 0, 0, "simulation")
    chunk = LiberoActionChunk(context, spec.action, ((0.,)*7,)*2, 1)
    queue.accept(chunk, context)
    assert queue.pop(context) == (0.,)*7
    with pytest.raises(ValueError, match="expiry"):
        queue.pop(SampleContext("e", 1, 50_000_000, "simulation"))
    assert len(queue) == 0
    queue.accept(chunk, context)
    queue.clear()
    assert queue.source is None
