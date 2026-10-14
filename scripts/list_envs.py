"""List this extension's registered tasks; no Kit launch or robot connections."""
import gymnasium as gym
import isaac_tasks  # noqa: F401


def main():
    tasks = [spec for spec in gym.registry.values()
             if str(spec.kwargs.get("env_cfg_entry_point", "")).startswith("isaac_tasks.")]
    for spec in sorted(tasks, key=lambda spec: spec.id):
        print(spec.id)
    if not tasks:
        print("No isaac-tasks environments registered: scaffold only.")


if __name__ == "__main__":
    main()
