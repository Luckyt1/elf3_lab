"""Floating-base GL-165, imported from URDF by Isaac Lab 2.1."""

import re
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.utils import clone
from pxr import Usd, UsdPhysics

from .model import DEFAULT_PHYSICS, JOINT_GROUPS, NESTED_COLLISION_PAIRS


@clone
def spawn_gl165(prim_path, cfg, translation=None, orientation=None):
    """Author mounting-pair exclusions before cloning the articulation."""
    prim = sim_utils.spawn_from_urdf(prim_path, cfg, translation, orientation)
    bodies = {p.GetName(): p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = ET.parse(cfg.asset_path).getroot().findall("joint")
    pairs = [(j.find("parent").get("link"), j.find("child").get("link")) for j in joints]
    for first, second in pairs + NESTED_COLLISION_PAIRS:
        if first not in bodies or second not in bodies:
            raise ValueError(f"Missing collision body: {first}, {second}")
        UsdPhysics.FilteredPairsAPI.Apply(bodies[first]).CreateFilteredPairsRel().AddTarget(bodies[second].GetPath())
    return prim


def make_gl165_cfg(physics=DEFAULT_PHYSICS) -> ArticulationCfg:
    # Called only when creating this task, so other tasks need no GL-165 files.
    if not physics.urdf.is_file():
        raise FileNotFoundError(physics.urdf)
    joints = {
        j.get("name"): j.find("limit")
        for j in ET.parse(physics.urdf).getroot().findall("joint")
        if j.get("type") == "revolute"
    }
    nominal = physics.default_joint_pos
    if joints.keys() != nominal.keys():
        raise ValueError("GL-165 XML-derived URDF does not match the configured 29 active joints.")
    actuators = {}
    for group, (pattern, kp, kd, _) in JOINT_GROUPS.items():
        names = [name for name in joints if re.fullmatch(pattern, name)]
        actuators[group] = ImplicitActuatorCfg(
            joint_names_expr=names,
            stiffness=kp,
            damping=kd,
            armature={name: physics.parameters["joint_parameters"][name]["armature_kg_m2"] for name in names},
            effort_limit_sim={name: float(joints[name].get("effort")) for name in names},
            velocity_limit_sim={name: float(joints[name].get("velocity")) for name in names},
        )
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            func=spawn_gl165,
            asset_path=str(physics.urdf),
            usd_dir=str(physics.directory / "usd"),
            usd_file_name="gl_165.usd",
            fix_base=False,
            root_link_name="base_link",
            # Keep the fixed head's individual masses, geometry and contact bodies.
            merge_fixed_joints=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            activate_contact_sensors=True,
            collider_type="convex_hull",
            # Retain self-contact except the explicit mechanical mounting pairs.
            self_collision=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, physics.parameters["stand_height_m"] + .003),
            joint_pos=nominal,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators=actuators,
    )
