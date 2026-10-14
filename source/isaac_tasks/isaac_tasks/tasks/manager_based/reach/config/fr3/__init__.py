"""Complete camera-equipped FR3 reaching environment registration."""
import gymnasium as gym

gym.register(id="Lab-VisualReach-FR3-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": f"{__name__}.visual_reach_env_cfg:Fr3VisualReachEnvCfg"})
