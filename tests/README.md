# tests

Reserved for isaac-tasks implementation. No hardware execution or task qualification is provided by this scaffold.
V1 offline tests cover physical expert increments, audited asset selection and
registration without loading assets. Run `python -m pytest tests --import-mode=importlib`.
The generic incomplete reach config remains deliberately invalid.

Simulator integration checks run through `scripts/run_visual_reach_expert.py`;
see `docs/v1_visual_reach.md`. They require the pinned Isaac environment, audited
external USD, compatible graphics driver and two cameras. No hardware is used.
