"""Manager-based reach configuration skeleton; intentionally not runnable."""
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING


@configclass
class ActionsCfg:
    """Define named action terms after the controller contract is selected."""


@configclass
class ObservationsCfg:
    """Define policy/critic groups with explicit feature order and frames."""


@configclass
class CommandsCfg:
    """Define goal sampling and success semantics."""


@configclass
class EventsCfg:
    """Define deterministic/randomized resets and physical randomization."""


@configclass
class RewardsCfg:
    """Define vectorized reward terms and documented units."""


@configclass
class TerminationsCfg:
    """Define timeouts separately from task failures."""


@configclass
class CurriculumCfg:
    """Define optional event/task curriculum terms."""


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    # Missing manager configs deliberately prevent a runnable empty task.
    scene: ReachSceneCfg = MISSING
    actions: ActionsCfg = MISSING
    observations: ObservationsCfg = MISSING
    commands: CommandsCfg = MISSING
    events: EventsCfg = MISSING
    rewards: RewardsCfg = MISSING
    terminations: TerminationsCfg = MISSING
    curriculum: CurriculumCfg = MISSING
