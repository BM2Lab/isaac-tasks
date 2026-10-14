"""V1 camera-equipped manager-based FR3 reaching task (no hardware equivalence claim)."""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp as official_mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaac_tasks.assets.fr3.visual_reach import make_robot_cfg
from isaac_tasks.controllers.visual_reach import CartesianIncrementActionCfg
from ...reach_env_cfg import ReachEnvCfg, ReachSceneCfg
from ...mdp import visual_reach as mdp


@configclass
class MountedCameraCfg(CameraCfg):
    mount_pos: tuple[float, float, float] = (0.14, 0, -0.03)
    mount_rot_ros_xyzw: tuple[float, float, float, float] = (0, -0.342020143326, 0, 0.939692620786)


def camera_cfg(prim_path, pos, rot, convention, mounted=False):
    cls = MountedCameraCfg if mounted else CameraCfg
    return cls(prim_path=prim_path, height=256, width=256, update_period=0,
        data_types=["rgb"], update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(focal_length=12, horizontal_aperture=20.955, clipping_range=(0.01, 3.0)),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention=convention))


@configclass
class VisualReachSceneCfg(ReachSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000))
    target_marker = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(radius=0.015,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1, 0.015, 0.015))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0, 0.45)))
    overhead = camera_cfg("{ENV_REGEX_NS}/OverheadCamera", (0.8, 0, 0.95),
        (-0.382683432365, 0, 0.923879532511, 0), "world")
    # Independent prim avoids Fabric overwriting a moving child-camera pose.
    # Observations compose its fixed mount with the measured flange transform.
    wrist = camera_cfg("{ENV_REGEX_NS}/WristCamera", (0.44689057, 0, 0.62028205),
        (-0.939692620786, 0, 0.342020143326, 0), "ros", mounted=True)
    wrist.spawn.focal_length = 8.0  # Wide enough for the initial approach and near-tip target.


@configclass
class VisualActionsCfg:
    cartesian_increment = CartesianIncrementActionCfg()


@configclass
class VisualObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_positions = ObservationTermCfg(func=mdp.joint_positions)
        overhead = ObservationTermCfg(func=mdp.rgb, params={"camera": "overhead"})
        wrist = ObservationTermCfg(func=mdp.rgb, params={"camera": "wrist"})
        concatenate_terms = False
        enable_corruption = False
    policy = PolicyCfg()


@configclass
class VisualCommandsCfg:
    target = mdp.VisibleTargetCommandCfg()


@configclass
class VisualEventsCfg:
    reset_robot = EventTermCfg(func=official_mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0, 0), "velocity_range": (0, 0)})
    reset_success = EventTermCfg(func=mdp.reset_success, mode="reset")


@configclass
class VisualRewardsCfg:
    distance = RewardTermCfg(func=mdp.distance_reward, weight=1)


@configclass
class VisualTerminationsCfg:
    success = TerminationTermCfg(func=mdp.sustained_success)
    invalid_state = TerminationTermCfg(func=mdp.invalid_state)
    time_out = TerminationTermCfg(func=official_mdp.time_out, time_out=True)


@configclass
class VisualCurriculumCfg:
    pass


@configclass
class Fr3VisualReachEnvCfg(ReachEnvCfg):
    scene = VisualReachSceneCfg(num_envs=1, env_spacing=2)
    actions = VisualActionsCfg()
    observations = VisualObservationsCfg()
    commands = VisualCommandsCfg()
    events = VisualEventsCfg()
    rewards = VisualRewardsCfg()
    terminations = VisualTerminationsCfg()
    curriculum = VisualCurriculumCfg()

    def __post_init__(self):
        self.scene.robot = make_robot_cfg()
        self.decimation = 10
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10
        self.num_rerenders_on_reset = 2
        self.viewer.eye = (1.2, 1.2, 1.0)
        self.viewer.lookat = (0.35, 0, 0.45)
