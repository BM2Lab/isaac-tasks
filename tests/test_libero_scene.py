"""Portable scene staging tests; physics qualification uses the separate CLI."""
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import pytest

from isaac_tasks.assets.panda.libero_scene import digest, load_reference, msh_to_obj, stage_components


def reference(tmp_path):
    source = tmp_path / "native"
    source.mkdir()
    (source/"model.xml").write_text('''<mujoco><asset/><worldbody>
    <body name="floor"><geom type="box" size="1 1 .1"/></body>
    <body name="robot0_base" pos="-0.6 0 0"><body name="link"/><body name="gripper0_eef"><site name="tip"/></body></body>
    <body name="alphabet_soup_1_main"><joint type="free" name="free"/></body>
    </worldbody><actuator/></mujoco>''')
    data = {"model": {"xml_sha256": digest(source/"model.xml"), "body_names": ["floor", "gripper0_eef"], "body_mass": [0, 0]},
            "environment": {"task_name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket"},
            "scene": {"body_pose_wxyz": {"robot0_base": [-.6, 0, 0, 1, 0, 0, 0],
                "alphabet_soup_1_main": [.1, .2, .3, 1, 0, 0, 0], "floor": [0, 0, 0, 1, 0, 0, 0]}}}
    (source/"reference.json").write_text(json.dumps(data))
    (source/"receipt.json").write_text(json.dumps({"reference_sha256": digest(source/"reference.json")}))
    return source


def test_native_receipt_and_no_overwrite(tmp_path):
    source = reference(tmp_path)
    stage_components(source, tmp_path/"out")
    with pytest.raises(FileExistsError):
        stage_components(source, tmp_path/"out")
    (source/"model.xml").write_text("changed")
    with pytest.raises(ValueError, match="MJCF checksum"):
        load_reference(source)


def test_split_keeps_root_pose_owned_by_init_state(tmp_path):
    source = reference(tmp_path)
    result = stage_components(source, tmp_path/"out")
    assert set(result["components"]) == {"robot", "room", "alphabet_soup_1_main"}
    root = ET.parse(tmp_path/"out/robot.xml").getroot()
    assert root.find("./worldbody/body").get("pos") == "0 0 0"
    assert root.find("actuator") is None
    assert len(root.findall("./worldbody/body")) == 1
    assert root.find(".//body[@name='gripper0_eef']") is None
    assert ET.parse(tmp_path/"out/room.xml").find(".//body[@name='floor']") is not None
    assert result["reference"]["scene"]["body_pose_wxyz"]["robot0_base"][0] == -.6


def test_mesh_conversion_and_bounds(tmp_path):
    source, target = tmp_path/"mesh.msh", tmp_path/"mesh.obj"
    source.write_bytes(struct.pack("<4i9f3i", 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2))
    msh_to_obj(source, target)
    assert "f 1 2 3" in target.read_text()
    assert sum(line.startswith("v ") for line in target.read_text().splitlines()) == 3
    source.write_bytes(struct.pack("<4i", -1, 0, 0, 0))
    with pytest.raises(ValueError):
        msh_to_obj(source, target)
