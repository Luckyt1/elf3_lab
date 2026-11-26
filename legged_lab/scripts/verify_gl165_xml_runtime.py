"""Verify the XML-derived GL-165 asset after Isaac/PhysX has instantiated it."""

import argparse
import json
import traceback
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arm-database", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from legged_lab.assets.gl_165.model import PHYSICS_DIR, PHYSICS_PARAMETERS, URDF_PATH
from legged_lab.envs.gl_165.stand_cfg import Gl165StandEnvCfg
from legged_lab.envs.gl_165.stand_env import Gl165StandEnv


def main():
    cfg = Gl165StandEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.seed = 42
    cfg.noise.add_noise = False
    cfg.push.enabled = False
    cfg.domain_rand.events.add_base_mass = None
    cfg.arm_motion.database = str(args.arm_database.resolve())
    env = Gl165StandEnv(cfg, headless=True)
    stage = Usd.Stage.Open(str(PHYSICS_DIR / "usd/gl_165.usd"))
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.MassAPI)}
    masses = env.robot.data.default_mass[0].cpu().numpy()
    inertias = env.robot.data.default_inertia[0].cpu().numpy().reshape(-1, 3, 3)
    centers = env.robot.root_physx_view.get_coms()[0, :, :3].cpu().numpy()
    rows = []
    for link in ET.parse(URDF_PATH).getroot().findall("link"):
        name = link.get("name")
        element = link.find("inertial")
        v = {k: float(x) for k, x in element.find("inertia").attrib.items()}
        expected = np.array([[v["ixx"], v["ixy"], v["ixz"]],
                             [v["ixy"], v["iyy"], v["iyz"]], [v["ixz"], v["iyz"], v["izz"]]])
        mass = UsdPhysics.MassAPI(bodies[name])
        q = mass.GetPrincipalAxesAttr().Get()
        axes = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
        actual = axes @ np.diag(np.array(mass.GetDiagonalInertiaAttr().Get())) @ axes.T
        index = env.robot.body_names.index(name)
        expected_mass = float(element.find("mass").get("value"))
        expected_center = np.fromstring(element.find("origin").get("xyz"), sep=" ")
        passed = (np.allclose(expected, actual, atol=1e-8, rtol=1e-4)
                  and np.allclose(expected, inertias[index], atol=1e-8, rtol=1e-4)
                  and np.allclose(expected_center, centers[index], atol=1e-6)
                  and np.isclose(masses[index], expected_mass, atol=1e-7, rtol=1e-5))
        rows.append({"body": name, "mass_kg": float(masses[index]), "passed": bool(passed),
                     "maximum_inertia_error": float(np.abs(expected - actual).max())})
    foot_spheres = {name: sum(1 for p in Usd.PrimRange(bodies[name], Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Sphere))
                    for name in PHYSICS_PARAMETERS["foot_contacts"]}
    foot_geometry = {name: [{"path": str(p.GetPath()), "type": p.GetTypeName()}
                           for p in Usd.PrimRange(bodies[name], Usd.TraverseInstanceProxies())
                           if p.HasAPI(UsdPhysics.CollisionAPI)]
                     for name in PHYSICS_PARAMETERS["foot_contacts"]}
    expected_limits = np.array([PHYSICS_PARAMETERS["joint_parameters"][n]["position_limits_rad"]
                                for n in env.robot.joint_names])
    middle = expected_limits.mean(axis=1)
    half_range = np.diff(expected_limits, axis=1)[:, 0] * .95 / 2
    expected_soft = np.stack((middle-half_range, middle+half_range), axis=1)
    soft_pass = bool(np.allclose(env.robot.data.soft_joint_pos_limits[0].cpu().numpy(), expected_soft, atol=1e-6))
    passed = all(r["passed"] for r in rows) and all(v == 9 for v in foot_spheres.values()) and soft_pass
    report = {"passed": passed, "bodies": rows, "body_count": len(rows), "foot_spheres": foot_spheres,
              "nominal_mass_kg": float(masses.sum()), "reward_terms": env.reward_manager.active_terms,
              "records_footprints": env.feet_anchor_xy is not None,
              "foot_geometry": foot_geometry,
              "soft_limits_match": soft_pass, "controller": env.controller_parameters,
              "actor_width": env.get_observations()[0].shape[-1]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "body_count", "foot_spheres", "soft_limits_match", "actor_width")}), flush=True)
    if not passed:
        raise RuntimeError("XML runtime audit failed; see the report")
    env.sim.clear_all_callbacks()
    env.sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = json.loads(args.output.read_text()) if args.output.exists() else {}
        report.update(passed=False, error=traceback.format_exc())
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        traceback.print_exc()
    finally:
        app.close(wait_for_replicator=False)
