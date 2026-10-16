"""P2 calibrated cameras/observation contract on the unchanged P1 scene."""
import json
import math
from pathlib import Path

import torch
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix
from isaac_tasks.assets.panda.libero_scene import load_reference, digest
from isaac_tasks.bridges.libero_visual import state_observation, rgb_observation
from .object_env_cfg import make_config


@configclass
class NativeCameraCfg(CameraCfg):
    mounted: bool = False
    mount_position: tuple = (0., 0., 0.)
    mount_rotation: tuple = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    reference_position: tuple = (0., 0., 0.)
    reference_rotation: tuple = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


@configclass
class VisualObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=state_observation)
        image = ObservationTermCfg(func=rgb_observation, params={"camera": "agentview"})
        image2 = ObservationTermCfg(func=rgb_observation, params={"camera": "eye_in_hand"})
        concatenate_terms = False
        enable_corruption = False
    policy = PolicyCfg()


def make_visual_config(scene_manifest, visual_reference, *, num_envs=1, device="cpu"):
    cfg = make_config(scene_manifest, num_envs=num_envs, device=device)
    ref = load_reference(visual_reference)
    manifest = json.loads(Path(scene_manifest).read_text())
    if not manifest["conversion"].get("native_visuals"):
        raise ValueError("P2 requires a cache rebuilt with --native-visuals")
    original = manifest["reference"]
    for component in manifest["components"].values():
        if component["kind"] == "static" and not component.get("visual_repairs", {}).get("render_only_floor_quads"):
            raise ValueError("Rebuild with the current P2 visual profile (floor UV mapping missing)")
        for filename, expected in component.get("visual_repairs", {}).get("external_textures_sha256", {}).items():
            if Path(filename).suffix.lower() not in (".png", ".jpg", ".jpeg") or digest(filename) != expected:
                raise ValueError("Pinned source texture changed")
    if (ref["model"]["xml_sha256"] != original["model"]["xml_sha256"]
            or ref["fixture"] != original["fixture"] or "visual" not in ref):
        raise ValueError("Visual and physics references must describe the exact same native scene/fixture")
    body = ref["scene"]["body_pose_wxyz"]["robot0_link7"]
    link_p = torch.tensor(body[:3], dtype=torch.float64)
    link_r = matrix_from_quat(torch.tensor(body[4:]+body[3:4], dtype=torch.float64))
    for source, name in (("agentview", "agentview"), ("robot0_eye_in_hand", "eye_in_hand")):
        camera = ref["visual"]["cameras"][source]
        if digest(Path(visual_reference)/camera["image"]) != camera["image_sha256"]:
            raise ValueError("Native reference image checksum mismatch")
        p = torch.tensor(camera["position_world"], dtype=torch.float64)
        r = torch.tensor(camera["rotation_world_opengl"], dtype=torch.float64)
        aperture = 2*24*math.tan(math.radians(camera["fovy_degrees"])/2)
        sensor = NativeCameraCfg(prim_path="{ENV_REGEX_NS}/Camera_"+name, height=256, width=256,
            update_period=0., data_types=["rgb"], update_latest_camera_pose=True,
            spawn=sim.PinholeCameraCfg(focal_length=24., vertical_aperture=aperture,
                horizontal_aperture=aperture, clipping_range=(.001, 100.)),
            offset=CameraCfg.OffsetCfg(pos=tuple(p.tolist()), rot=tuple(quat_from_matrix(r).tolist()), convention="opengl"))
        sensor.mounted = name == "eye_in_hand"
        sensor.mount_position = tuple((link_r.T@(p-link_p)).tolist())
        sensor.mount_rotation = tuple(tuple(row) for row in (link_r.T@r).tolist())
        sensor.reference_position = tuple(p.tolist())
        sensor.reference_rotation = tuple(tuple(row) for row in r.tolist())
        setattr(cfg.scene, name, sensor)
    cfg.observations = VisualObservationsCfg()
    cfg.libero_orientation_chain = ref["visual"]["orientation_chain"]
    cfg.scene.light.spawn.intensity = 300.  # RTX exposure calibration, not native GL lighting parity.
    cfg.num_rerenders_on_reset = 2
    return cfg
