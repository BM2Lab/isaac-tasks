"""Native soup/basket scene with explicit imported components and OSC action."""
import json
from pathlib import Path

import isaaclab.sim as sim
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaac_tasks.assets.panda.libero_scene import digest
from isaac_tasks.controllers.libero_osc_action import LiberoOscActionCfg


@configclass
class ActionsCfg:
    osc = LiberoOscActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class StateCfg(ObservationGroupCfg):
        q = ObservationTermCfg(func=mdp.joint_pos)
        dq = ObservationTermCfg(func=mdp.joint_vel)
        enable_corruption = False
    state = StateCfg()


@configclass
class EventsCfg:
    reset = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


def make_config(manifest_path, *, num_envs=1, device="cpu"):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest.get("generated_files_sha256", {}).items():
        candidate = (manifest_path.parent/name).resolve()
        if not candidate.is_relative_to(manifest_path.parent) or digest(candidate) != expected:
            raise ValueError(f"Converted cache dependency changed: {name}")
    ref = manifest["reference"]
    scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=3.0, replicate_physics=True)
    poses = ref["scene"]["body_pose_wxyz"]
    def initial(name):
        pose = poses[name]
        return {"pos": tuple(pose[:3]), "rot": tuple(pose[4:]+pose[3:4])}  # XYZW
    for name, component in manifest["components"].items():
        usd = manifest_path.parent/component["usd"]
        if digest(usd) != component["usd_sha256"]:
            raise ValueError(f"Converted asset changed: {name}")
        prim = "{ENV_REGEX_NS}/"+name
        spawn = sim.UsdFileCfg(usd_path=str(usd))
        if name == "robot":
            joint_positions = {key: value[0] for key, value in ref["scene"]["joint_qpos"].items()
                               if key.startswith(("robot0_joint", "gripper0_finger_joint"))}
            joint_velocities = {key: value[0] for key, value in ref["scene"]["joint_qvel"].items()
                               if key in joint_positions}
            actuators = {}
            for i in range(7):
                actuators[f"arm_{i}"] = ImplicitActuatorCfg(joint_names_expr=[f"robot0_joint{i+1}"],
                    stiffness=0., damping=0., effort_limit_sim=1e6,
                    friction=0., dynamic_friction=0., viscous_friction=0.,
                    armature=ref["scene"]["dof_armature"][ref["model"]["arm_qvel_indices"][i]])
            actuators["fingers"] = ImplicitActuatorCfg(joint_names_expr=["gripper0_finger_joint.*"],
                stiffness=0., damping=0., effort_limit_sim=1e6, armature=1.,
                friction=0., dynamic_friction=0., viscous_friction=0.)
            scene.robot = ArticulationCfg(prim_path=prim, spawn=spawn,
                init_state=ArticulationCfg.InitialStateCfg(**initial("robot0_base"),
                    joint_pos=joint_positions, joint_vel=joint_velocities), actuators=actuators,
                soft_joint_pos_limit_factor=1.)
        elif component["kind"] == "static":
            setattr(scene, name, AssetBaseCfg(prim_path=prim, spawn=spawn))
        else:
            setattr(scene, name, RigidObjectCfg(prim_path=prim, spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(**initial(name))))
    scene.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim.DomeLightCfg(intensity=1500))
    cfg = ManagerBasedEnvCfg(scene=scene, actions=ActionsCfg(), observations=ObservationsCfg(),
                            events=EventsCfg(), decimation=25)
    cfg.actions.osc.scene_manifest = str(manifest_path)
    cfg.sim.device = device
    cfg.seed = 42
    cfg.sim.dt = .002
    cfg.sim.render_interval = 25
    cfg.viewer.eye = (1.3, 1.3, 1.3)
    cfg.viewer.lookat = (0., 0., .3)
    return cfg
