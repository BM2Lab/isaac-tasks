"""Import checks use the pinned Isaac environment; no Kit launch."""
import pytest

from isaac_tasks.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg


def test_reach_requires_configuration():
    # Isaac's configclass deep-copies defaults; validate semantics, not identity.
    with pytest.raises(TypeError, match="Missing values"):
        ReachEnvCfg().validate()
