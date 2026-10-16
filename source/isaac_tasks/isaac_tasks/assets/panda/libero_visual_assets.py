"""P2 render-only repairs to importer output. Never edits collision enablement."""
from pathlib import Path
import xml.etree.ElementTree as ET


def repair_visuals(stage, mjcf):
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    from .libero_scene import digest
    root = ET.parse(mjcf).getroot()
    geom_groups = {x.get("name"): int(x.get("group", 0)) for x in root.findall(".//geom")}
    textures = {x.get("name"): x for x in root.findall("./asset/texture")}
    materials = {x.get("name"): x for x in root.findall("./asset/material")}
    dependencies = {}
    hidden = []
    for prim in stage.Traverse():
        # MuJoCo's default viewer renders group 1 visual geometry, not debug
        # group 0 collision meshes or any site visualizers.
        if (prim.GetName() in geom_groups and geom_groups[prim.GetName()] == 0
                or "MjcSiteAPI" in prim.GetAppliedSchemas()):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(str(prim.GetPath()))
        if not prim.IsA(UsdShade.Material) or prim.GetName() not in materials:
            continue
        definition = materials[prim.GetName()]
        material = UsdShade.Material(prim)
        # Flattened importer materials reference shader prototypes OUTSIDE the
        # default prim, and textures in a now-deleted importer temp directory.
        # Replace that shading network using original pinned source assets.
        prim.SetInstanceable(False)
        prim.GetReferences().ClearReferences()
        surface = UsdShade.Shader.Define(stage, prim.GetPath().AppendChild("NativeSurface"))
        surface.CreateIdAttr("UsdPreviewSurface")
        color = [float(x) for x in definition.get("rgba", "1 1 1 1").split()]
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1-float(definition.get("shininess", ".1")))
        surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(color[3])
        texture = textures.get(definition.get("texture"))
        if texture is not None and texture.get("file"):
            # Preserve the snapshot filename extension. HF snapshot PNG paths
            # are symlinks to extensionless blobs; RTX cannot infer their image
            # type if we dereference that filename into the blob path.
            path = Path(texture.get("file")).absolute()
            if not path.is_file():
                raise FileNotFoundError(path)
            dependencies[str(path)] = digest(path)
            uv = UsdShade.Shader.Define(stage, prim.GetPath().AppendChild("NativeUV"))
            uv.CreateIdAttr("UsdPrimvarReader_float2")
            uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            image = UsdShade.Shader.Define(stage, prim.GetPath().AppendChild("NativeTexture"))
            image.CreateIdAttr("UsdUVTexture")
            image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
            image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            image.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            image.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
            image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surface.GetInput("diffuseColor").ConnectToSource(image.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.GetDisplacementOutput().DisconnectSource()
    # USD Plane has no authored UVs. Native MuJoCo texuniform projection must
    # be represented by a render-only quad; retain the original plane collider.
    floor_visuals = []
    for prim in list(stage.Traverse()):
        if prim.GetTypeName() != "Plane":
            continue
        definition = next((x for x in root.findall(".//geom") if x.get("name") == prim.GetName()), None)
        if definition is None or definition.get("material") not in materials:
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        width = float(prim.GetAttribute("width").Get())/2
        length = float(prim.GetAttribute("length").Get())/2
        mesh = UsdGeom.Mesh.Define(stage, prim.GetPath().GetParentPath().AppendChild(prim.GetName()+"_visual"))
        mesh.CreatePointsAttr([(-width, -length, 0), (width, -length, 0), (width, length, 0), (-width, length, 0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        repeat = [float(x) for x in materials[definition.get("material")].get("texrepeat", "1 1").split()]
        uv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        uv.Set([(-width*repeat[0], -length*repeat[1]), (width*repeat[0], -length*repeat[1]),
                (width*repeat[0], length*repeat[1]), (-width*repeat[0], length*repeat[1])])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        UsdGeom.Imageable(prim).MakeInvisible()
        floor_visuals.append(str(mesh.GetPath()))
    return {"hidden_debug_geometry": hidden, "external_textures_sha256": dependencies,
            "render_only_floor_quads": floor_visuals}
