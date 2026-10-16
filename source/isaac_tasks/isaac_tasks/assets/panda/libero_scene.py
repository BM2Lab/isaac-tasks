"""Pinned native MJCF staging and receipts; no MuJoCo/Isaac imports here.

Generated meshes/USDs belong in an explicit conversion cache, never in Git.
Splits the native scene into one fixed Panda, static room, and free objects.
"""
import copy
from hashlib import sha256
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_reference(directory):
    directory = Path(directory).resolve()
    reference = json.loads((directory / "reference.json").read_text())
    receipt = json.loads((directory / "receipt.json").read_text())
    if digest(directory / "reference.json") != receipt["reference_sha256"]:
        raise ValueError("Reference checksum mismatch")
    if digest(directory / "model.xml") != reference["model"]["xml_sha256"]:
        raise ValueError("MJCF checksum mismatch")
    if "scene" not in reference:
        raise ValueError("Re-export native reference with P1 scene metadata")
    if reference["environment"]["task_name"] != "pick_up_the_alphabet_soup_and_place_it_in_the_basket":
        raise ValueError("P1 supports only the soup-in-basket scene")
    return reference


def msh_to_obj(source, destination):
    """MuJoCo binary mesh: counts, vertices, normals, UVs, triangle indices.

    Face vertex/normal/UV indices share vertex indexing in this legacy format.
    Keep material/texture references in the MJCF; OBJ contains geometry only.
    """
    data = Path(source).read_bytes()
    nv, nn, nt, nf = struct.unpack_from("<4i", data)
    if min(nv, nn, nt, nf) < 0 or nv == 0 or nf == 0:
        raise ValueError("Invalid binary mesh counts")
    expected = 16+12*nv+12*nn+8*nt+12*nf
    if len(data) != expected or nn not in (0, nv) or nt not in (0, nv):
        raise ValueError("Unsupported binary mesh layout")
    offset = 16
    lines = []
    for prefix, count, width in (("v", nv, 3), ("vn", nn, 3), ("vt", nt, 2)):
        values = struct.unpack_from(f"<{count*width}f", data, offset)
        offset += count*width*4
        lines.extend(prefix+" "+" ".join(format(x, ".9g") for x in values[i:i+width])
                     for i in range(0, len(values), width))
    faces = struct.unpack_from(f"<{nf*3}i", data, offset)
    if any(i < 0 or i >= nv for i in faces):
        raise ValueError("Mesh face index outside vertex array")
    def index(i):
        v = str(i+1)
        return v+("/"+v if nt else "/")+("/"+v if nn else "") if nt or nn else v
    lines.extend("f "+" ".join(index(i) for i in faces[k:k+3]) for k in range(0, len(faces), 3))
    Path(destination).write_text("\n".join(lines)+"\n")


def stage_components(reference_dir, output):
    reference_dir, output = Path(reference_dir).resolve(), Path(output).resolve()
    ref = load_reference(reference_dir)
    output.mkdir(parents=True, exist_ok=False)
    root = ET.parse(reference_dir / "model.xml").getroot()
    mesh_receipts = {}
    for element in root.findall("./asset/*[@file]"):
        source = Path(element.attrib["file"])
        if not source.is_absolute():
            raise ValueError("Native export must resolve all asset paths")
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix == ".msh":
            target = output / (digest(source)+".obj")
            if not target.exists():
                msh_to_obj(source, target)
            element.set("file", str(target))
            mesh_receipts[str(source)] = {"source_sha256": digest(source), "obj_sha256": digest(target)}
    world = root.find("worldbody")
    robot = next(body for body in world.findall("body") if body.get("name") == "robot0_base")
    free = [body for body in world.findall("body") if body.find("joint[@type='free']") is not None]
    components = {"robot": [robot], "room": [x for x in world if x is not robot and x not in free]}
    components.update({body.get("name"): [body] for body in free})
    manifest = {"schema_version": 1, "reference_sha256": digest(reference_dir / "reference.json"),
                "staging_source_sha256": digest(__file__),
                "mesh_conversion": mesh_receipts, "components": {}, "reference": ref}
    for name, elements in components.items():
        document = copy.deepcopy(root)
        for tag in ("actuator", "sensor", "equality", "contact", "keyframe", "tendon"):
            for item in document.findall(tag):
                document.remove(item)
        body_root = document.find("worldbody")
        body_root.clear()
        for element in elements:
            item = copy.deepcopy(element)
            if item.tag == "body" and item.get("name") in ref["scene"]["body_pose_wxyz"]:
                pose = ref["scene"]["body_pose_wxyz"][item.get("name")]
                # Isaac asset init_state owns movable root poses. Keep the
                # converted root identity to avoid applying the pose twice.
                item.set("pos", "0 0 0" if name != "room" else " ".join(map(str, pose[:3])))
                item.set("quat", "1 0 0 0" if name != "room" else " ".join(map(str, pose[3:])))
            body_root.append(item)
        # PhysX cannot represent a separate zero-mass dynamic link: importer
        # gives such a frame default mass. Drop massless leaf frames (sites are
        # reconstructed from the exported transforms by the adapter instead).
        zero_mass = {name for name, mass in zip(ref["model"].get("body_names", []),
                    ref["model"].get("body_mass", [])) if mass == 0}
        for parent in list(body_root.iter()):
            for item in list(parent):
                if (name == "robot" and item.tag == "body" and item.get("name") in zero_mass
                        and item.find("body") is None and item.find("joint") is None):
                    parent.remove(item)
        # Drop unused assets, including expensive room meshes from robot files.
        used = {x.get(key) for x in body_root.iter() for key in ("mesh", "material") if x.get(key)}
        asset = document.find("asset")
        for material in asset.findall("material"):
            if material.get("name") in used and material.get("texture"):
                used.add(material.get("texture"))
        for item in list(asset):
            if item.get("name") not in used:
                asset.remove(item)
        path = output / f"{name}.xml"
        ET.ElementTree(document).write(path, encoding="unicode")
        manifest["components"][name] = {"xml": path.name, "sha256": digest(path),
            "kind": "articulation" if name == "robot" else "static" if name == "room" else "rigid"}
    (output / "staged.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest
