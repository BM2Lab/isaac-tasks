"""Single-env synchronous simulation session; no ML imports and no real-time guarantees."""
import base64
import uuid
import hashlib
import importlib.metadata
import sys

import torch
from contracts.actions import ActionChunk
from contracts.episodes import SampleContext
from contracts.samples import ObservationPacket, RGBSample
from .action_queue import ActionQueue
from .visual_reach import VisualReachBridge
from isaac_tasks.assets.fr3.visual_reach import MODEL


class LocalVisualReachSession:
    def __init__(self, env, spec, seed=42):
        if env.num_envs != 1:
            raise ValueError("V2 local sessions support exactly one environment")
        self.env, self.spec = env, spec
        self.bridge = VisualReachBridge(env, spec)
        self.queue = ActionQueue(spec)
        self.session_id = uuid.uuid4().hex
        self.episode_number = -1
        self.total_steps, self.step_index = 0, 0
        self.period_ns = round(env.step_dt*1e9)
        self.reset(seed)

    @property
    def context(self):
        return SampleContext(f"{self.session_id}:{self.episode_number}", self.step_index,
            self.total_steps*self.period_ns, "simulation")

    def _new_episode(self):
        self.queue.clear()
        self.episode_number += 1
        self.step_index = 0

    def reset(self, seed=None):
        self._new_episode()
        try:
            self.observations, _ = self.env.reset(seed=seed)
        except Exception as exc:
            raise RuntimeError("Simulation reset failed; this session must be restarted") from exc
        return self.observe()

    def receipt(self):
        return {"session_id": self.session_id, "backend": "simulation",
            "contract_sha256": hashlib.sha256(self.spec.to_json().encode()).hexdigest(),
            "asset": MODEL, "physics_dt_s": self.env.physics_dt, "policy_dt_s": self.env.step_dt,
            "episode_length_s": self.env.cfg.episode_length_s,
            "runtime": {"python": sys.version, **{name: importlib.metadata.version(name)
                for name in ("isaaclab", "isaacsim", "torch", "contracts", "isaac-tasks")}}}

    def observe(self):
        context = self.context
        features = self.bridge.policy_features(self.observations)
        images = []
        for camera in self.spec.observation.cameras:
            sensor = self.env.scene[self.bridge.cameras[camera.key]]
            rgb = features[camera.key][0].contiguous().cpu().numpy().tobytes()
            images.append(RGBSample(camera.key, camera.optical_frame, camera.height, camera.width,
                int(sensor.frame.torch[0]), context, base64.b64encode(rgb).decode("ascii")))
        packet = ObservationPacket(self.spec.contract_id, context,
            tuple(features[self.spec.observation.state_key][0].cpu().tolist()), tuple(images),
            features[self.spec.observation.instruction_key])
        packet.validate_for(self.spec)
        return packet.to_dict()

    def submit(self, chunk):
        self.queue.accept(ActionChunk.from_dict(chunk), self.context)
        return self.status()

    def advance(self, expected_context):
        if SampleContext.from_dict(expected_context) != self.context:
            raise ValueError("Stale advance request; simulation has not advanced")
        row = self.queue.pop(self.context)
        action = torch.tensor([row], device=self.env.device, dtype=torch.float32)
        try:
            self.observations, _, terminated, truncated, _ = self.env.step(action)
        except Exception as exc:
            self.queue.clear()
            raise RuntimeError("Physics step failed; this session must be restarted") from exc
        self.total_steps += 1
        self.step_index += 1
        done = bool(terminated[0] | truncated[0])
        terminal = None
        if done:
            terminal = {"episode_id": self.context.episode_id,
                "success": bool(self.env.termination_manager.get_term("success")[0]),
                "invalid_state": bool(self.env.termination_manager.get_term("invalid_state")[0]),
                "timeout": bool(truncated[0]), "position_error_m": float(self.env.reach_terminal_error[0])}
            # ManagerBasedRLEnv has already reset physics and cameras. New sample
            # identity must match those returned observations, not the old episode.
            self._new_episode()
        return {"observation": self.observe(), "terminal": terminal, "status": self.status()}

    def status(self):
        return {"context": self.context.to_dict(), "queued_actions": len(self.queue), "backend": "simulation"}
