"""Audited external FR3 asset; no network downloads or legacy package imports."""
import hashlib
import json
import os
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

MODEL = json.loads(Path(__file__).with_name("model.json").read_text())


def make_robot_cfg():
    value = os.environ.get("ISAAC_TASKS_FR3_USD")
    if not value:
        raise RuntimeError("Set ISAAC_TASKS_FR3_USD to the audited FR3 bare-flange USD (see assets/fr3/README.md)")
    path = Path(value).expanduser().resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL["usd_sha256"]:
        raise ValueError("FR3 USD checksum differs from the V1 audited model")
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=dict(zip(MODEL["joint_names"], MODEL["home_rad"])), joint_vel={".*": 0.0}),
        actuators={
            "shoulder": ImplicitActuatorCfg(joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0, stiffness=400.0, damping=80.0, armature=0.003),
            "wrist": ImplicitActuatorCfg(joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0, stiffness=400.0, damping=80.0, armature=0.003)},
        soft_joint_pos_limit_factor=1.0)
