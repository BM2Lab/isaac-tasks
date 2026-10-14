"""Task discovery, matching Isaac Lab's external extension workflow."""
from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp", "common_mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
