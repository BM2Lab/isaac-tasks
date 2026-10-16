"""Convert the pinned P1 native scene into an explicit USD build cache."""
import argparse
import json
from pathlib import Path
import shutil
import traceback
import importlib.metadata

from isaaclab.app import AppLauncher
from isaac_tasks.assets.panda.libero_scene import stage_components, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 2*1024**3:
        raise RuntimeError("Existing conversion cache parent with 2 GiB free required")
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Converted assets must remain outside source")
    manifest = stage_components(args.reference_dir, output)
    launcher = AppLauncher(args)
    status = 1
    try:
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
        for name, component in manifest["components"].items():
            result = MjcfConverter(MjcfConverterCfg(asset_path=str(output/component["xml"]),
                usd_dir=str(output/name), force_usd_conversion=True,
                fix_base=component["kind"] == "articulation", self_collision=True,
                run_asset_transformer=False, run_multi_physics_conversion=True))
            # The importer marks jointless MJCF world bodies as dynamic USD
            # bodies. Native room/floor are static; dynamic triangle planes
            # are invalid in PhysX. Author the correction in this cache only.
            from pxr import Usd, UsdPhysics
            stage = Usd.Stage.Open(result.usd_path)
            for prim in stage.Traverse():
                if component["kind"] == "static" and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI) and component["kind"] != "static":
                    UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("convexHull")
                # MuJoCo frictionloss is an absolute effort, not PhysX's
                # deprecated load-proportional joint friction coefficient.
                friction = prim.GetAttribute("physxJoint:jointFriction")
                if friction and component["kind"] == "articulation":
                    friction.Set(0.)
                if component["kind"] == "articulation" and prim.GetName() == "robot0_base":
                    # This link is fixed to world. Positive dummy inertia avoids
                    # PhysX's invalid-zero-mass fallback; it cannot affect M/J/b
                    # of the downstream fixed-base arm (checked independently).
                    from pxr import Gf
                    mass = UsdPhysics.MassAPI.Apply(prim)
                    mass.CreateMassAttr(1.)
                    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001))
            stage.GetRootLayer().Save()
            component["usd"] = str(Path(result.usd_path).relative_to(output))
            component["usd_sha256"] = digest(result.usd_path)
            print(name, result.usd_path, flush=True)
        manifest["conversion"] = {"self_collision": True, "collision_from_visuals": False,
            "source_sha256": digest(__file__),
            "run_asset_transformer": False, "run_multi_physics_conversion": True,
            "massless_leaf_frames": "omitted; TCP reconstructed from native receipt",
            "fixed_base_dummy_mass_kg": 1., "legacy_joint_friction_coefficient": 0.,
            "runtime": {name: importlib.metadata.version(name) for name in ("isaacsim", "isaaclab")},
            "note": "MuJoCo contact/solver semantics are not exactly representable in PhysX"}
        manifest["generated_files_sha256"] = {str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*")) if p.is_file()}
        (output/"scene.json").write_text(json.dumps(manifest, indent=2)+"\n")
        status = 0
        print("Scene cache:", output/"scene.json", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if launcher.app is not None:
            launcher.app.close(exit_code=status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
