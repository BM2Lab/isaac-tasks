"""Paused synchronous P2 simulation session. No policy or hardware authority."""
import base64
from hashlib import sha256
import json
from pathlib import Path
import uuid
import importlib.metadata
import inspect

import torch
from contracts.episodes import SampleContext
from contracts.libero import LiberoActionChunk, LiberoIOSpec
from contracts.samples import ObservationPacket, RGBSample
from .action_queue import ActionQueue


class LocalLiberoSession:
    def __init__(self, env, visual_reference, *, max_steps=280):
        if env.num_envs != 1 or env.step_dt != .05 or type(max_steps) is not int or not 1 <= max_steps <= 280:
            raise ValueError("P2 requires one 20 Hz environment and a bounded horizon")
        self.env, self.spec = env, LiberoIOSpec()
        self.reference = json.loads((Path(visual_reference)/"reference.json").read_text())
        self.reference_path = Path(visual_reference)/"reference.json"
        self.max_steps = max_steps
        self.queue = ActionQueue(self.spec)
        self.session_id = uuid.uuid4().hex
        self.episode = -1
        self.total_steps = 0
        self.failed = False
        self.reset()

    @property
    def context(self):
        return SampleContext(f"{self.session_id}:{self.episode}", self.step,
                             self.total_steps*50_000_000, "simulation")

    def _capture(self):
        features = self.observations["policy"]
        context = self.context
        images = []
        for camera, short in zip(self.spec.observation.cameras, ("agentview", "eye_in_hand")):
            key = "image" if short == "agentview" else "image2"
            image = features[key][0]
            if image.shape != (256, 256, 3) or image.dtype != torch.uint8:
                raise RuntimeError("Wrong native camera tensor")
            raw = image.contiguous().cpu().numpy().tobytes()
            images.append(RGBSample(camera.key, camera.optical_frame, 256, 256,
                int(self.env.scene[short].frame.torch[0]), context, base64.b64encode(raw).decode("ascii")))
        packet = ObservationPacket(self.spec.contract_id, context, tuple(features["state"][0].cpu().tolist()),
            tuple(images), self.reference["environment"]["instruction"])
        packet.validate_for(self.spec)
        self.packet = packet.to_dict()

    def reset(self, seed=None):
        if self.failed:
            raise RuntimeError("Session is faulted; restart the simulator")
        self.queue.clear()
        self.episode += 1
        self.step = 0
        self.done = False
        # This task replays the exported fixed fixture, not a random seed sweep.
        try:
            self.observations, _ = self.env.reset(seed=seed)
            self._capture()
        except Exception as exc:
            self.done = True
            self.failed = True
            raise RuntimeError("LIBERO reset failed; restart the simulator session") from exc
        return self.observe()

    def observe(self):
        # A stable capture identity. Repeated observes must not render/step or
        # attach a new frame to an old source-action context.
        return self.packet

    def status(self):
        return {"context": self.context.to_dict(), "queued_actions": len(self.queue),
                "backend": "simulation", "done": self.done}

    def receipt(self):
        from isaac_tasks.assets.panda.libero_scene import digest
        from isaac_tasks.bridges import libero_visual
        return {"backend": "isaac_physx_libero_port_not_native_mujoco", "session_id": self.session_id,
                "contract_sha256": sha256(self.spec.to_json().encode()).hexdigest(),
                "fixture_sha256": self.reference["fixture"]["sha256"],
                "visual_reference_sha256": digest(self.reference_path),
                "scene_manifest_sha256": digest(self.env.cfg.actions.osc.scene_manifest),
                "runtime": {name: importlib.metadata.version(name) for name in ("isaaclab", "isaacsim", "torch", "contracts")},
                "source_sha256": {Path(p).name: digest(p) for p in (__file__, inspect.getfile(libero_visual),
                    inspect.getfile(type(self.env.action_manager.get_term("osc"))))},
                "policy_dt_s": self.env.step_dt, "physics_dt_s": self.env.physics_dt,
                "task_success_implemented": False, "image_transform_from_isaac": "horizontal_flip_once",
                "rendering_equivalence": "camera_geometry_matched_not_pixel_identical",
                "max_steps": self.max_steps}

    def submit(self, chunk):
        if self.done:
            raise ValueError("Episode ended; reset before submitting")
        chunk = LiberoActionChunk.from_dict(chunk)
        if self.step+len(chunk.values) > self.max_steps:
            raise ValueError("Chunk exceeds remaining episode budget")
        self.queue.accept(chunk, self.context)
        return self.status()

    def expert_action(self, context):
        raise ValueError("No privileged expert implemented for P2")

    def advance(self, expected_context):
        if self.done or SampleContext.from_dict(expected_context) != self.context:
            raise ValueError("Ended episode or stale advance context; no physics advanced")
        row = self.queue.pop(self.context)
        try:
            self.observations, _ = self.env.step(torch.tensor([row], dtype=torch.float32, device=self.env.device))
            self.total_steps += 1
            self.step += 1
            self._capture()
        except Exception as exc:
            self.queue.clear()
            self.done = True
            self.failed = True
            raise RuntimeError("LIBERO step failed; restart this session") from exc
        self.done = self.step == self.max_steps
        if self.done:
            self.queue.clear()
        terminal = {"reason": "horizon", "success": None} if self.done else None
        return {"observation": self.observe(), "terminal": terminal, "status": self.status()}
