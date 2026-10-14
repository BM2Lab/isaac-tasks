"""Single-env synchronous simulation session; no ML imports and no real-time guarantees."""
import base64
import uuid
import hashlib
import importlib.metadata
import sys
import inspect
from pathlib import Path

import torch
from contracts.actions import ActionChunk
from contracts.episodes import SampleContext
from contracts.samples import ObservationPacket, RGBSample
from .action_queue import ActionQueue
from .visual_reach import VisualReachBridge
from isaac_tasks.assets.fr3.visual_reach import MODEL


class LocalVisualReachSession:
    def __init__(self, env, spec, seed=42, enable_expert=False):
        if env.num_envs != 1:
            raise ValueError("V2 local sessions support exactly one environment")
        self.env, self.spec = env, spec
        self.enable_expert = enable_expert
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
        from isaac_tasks.controllers.visual_reach import CartesianIncrementAction, expert_increment
        from isaac_tasks.tasks.manager_based.reach.mdp.visual_reach import sustained_success
        sources = {"environment": type(self.env.cfg), "controller_and_expert": CartesianIncrementAction,
            "target_and_termination_terms": sustained_success}
        return {"session_id": self.session_id, "backend": "simulation",
            "contract_sha256": hashlib.sha256(self.spec.to_json().encode()).hexdigest(),
            "asset": MODEL, "physics_dt_s": self.env.physics_dt, "policy_dt_s": self.env.step_dt,
            "episode_length_s": self.env.cfg.episode_length_s,
            "expert_demonstrations_enabled": self.enable_expert,
            "source_sha256": {name: hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest() for name, obj in sources.items()},
            "task_recipe": {"target_lower_m": self.env.cfg.commands.target.lower,
                "target_upper_m": self.env.cfg.commands.target.upper,
                "implicit_pd": {name: {"stiffness": cfg.stiffness, "damping": cfg.damping,
                    "armature": cfg.armature, "effort_limit_sim": cfg.effort_limit_sim}
                    for name, cfg in self.env.scene["robot"].cfg.actuators.items()},
                "expert_position_gain": inspect.signature(expert_increment).parameters["gain"].default,
                "expert_max_translation_norm_m": self.spec.action.max_translation_norm_m-1e-8},
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

    def expert_action(self, expected_context):
        """Opt-in privileged label generation, never a policy observation feature."""
        if not self.enable_expert:
            raise ValueError("Expert demonstrations are disabled for this session")
        if SampleContext.from_dict(expected_context) != self.context or len(self.queue):
            raise ValueError("Expert label requires current capture context and an empty queue")
        from isaac_tasks.controllers.visual_reach import tcp_pose_base, expert_increment
        row = expert_increment(tcp_pose_base(self.env)[0], self.env.command_manager.get_command("target"),
            max_norm_m=self.spec.action.max_translation_norm_m-1e-8)[0].cpu().tolist()
        chunk = ActionChunk(self.context, self.spec.action, (tuple(row),), self.context.timestamp_ns+2*self.period_ns)
        return {"chunk": chunk.to_dict(), "label_source": "privileged_scripted_ik_expert"}

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
