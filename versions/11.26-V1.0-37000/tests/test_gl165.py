"""Asset and disturbance checks; run using python -m unittest discover -s tests -v."""

import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = load_file("gl165_model", "legged_lab/assets/gl_165/model.py")
pushes = load_file("gl165_pushes", "legged_lab/envs/gl_165/pushes.py")
stance = load_file("gl165_stance", "legged_lab/assets/gl_165/stance_recovery.py")
rewards = load_file("gl165_rewards", "legged_lab/envs/gl_165/rewards.py")
prepare = load_file("gl165_prepare", "legged_lab/scripts/prepare_gl165.py")
arm_motion = load_file("gl165_arm_motion", "legged_lab/envs/gl_165/arm_motion.py")
controller_audit = load_file("gl165_controller_audit", "legged_lab/scripts/audit_gl165_controller.py")


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.root = ET.parse(model.URDF_PATH).getroot()

    def test_updated_cad_dynamics_and_meshes(self):
        import hashlib
        physics = model.PHYSICS_PARAMETERS
        source = model.PHYSICS_DIR / physics["source_urdf_copy"]
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), physics["source_urdf_sha256"])
        links = {link.get("name"): link for link in self.root.findall("link")}
        for cad in ET.parse(source).findall("link[inertial]"):
            link = links[physics["cad_link_to_training_link"][cad.get("name")]]
            for tag in ("mass", "inertia", "origin"):
                self.assertEqual(link.find(f"inertial/{tag}").attrib, cad.find(f"inertial/{tag}").attrib)
            mesh = Path(link.find("visual/geometry/mesh").get("filename"))
            self.assertEqual(mesh.name, Path(cad.find("visual/geometry/mesh").get("filename")).name)
            self.assertEqual(hashlib.sha256((model.URDF_PATH.parent / mesh).read_bytes()).hexdigest(),
                             physics["source_mesh_sha256"][mesh.name])
        legacy = model.load_physics(model.LEGACY_PHYSICS_PROFILE)
        self.assertNotEqual(physics["source_urdf_sha256"], legacy.parameters["source_urdf_sha256"])
        self.assertAlmostEqual(physics["mass_kg"], 43.22284735813295)

    def test_floating_tree_and_mesh_paths(self):
        links = {x.get("name") for x in self.root.findall("link")}
        children = {x.find("child").get("link") for x in self.root.findall("joint")}
        self.assertEqual(links - children, {"base_link"})
        self.assertNotIn("ground_link", links)
        self.assertEqual(len(links), 32)
        for mesh in self.root.iter("mesh"):
            self.assertTrue((model.URDF_PATH.parent / mesh.get("filename")).is_file())

    def test_joint_control_coverage_and_pose_limits(self):
        joints = self.root.findall("joint[@type='revolute']")
        self.assertEqual({j.get("name") for j in joints}, set(model.DEFAULT_JOINT_POS))
        for joint in joints:
            name = joint.get("name")
            groups = [v for v in model.JOINT_GROUPS.values() if re.fullmatch(v[0], name)]
            self.assertEqual(len(groups), 1, name)
            limit = joint.find("limit")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            margin = (upper - lower) * 0.025
            self.assertGreaterEqual(model.DEFAULT_JOINT_POS[name], lower + margin, name)
            self.assertLessEqual(model.DEFAULT_JOINT_POS[name], upper - margin, name)
            self.assertGreater(float(limit.get("effort")), 0.0)
        policy = {name for name in model.DEFAULT_JOINT_POS if re.fullmatch(model.POLICY_JOINT_PATTERN, name)}
        self.assertEqual(len(policy), 15)
        self.assertFalse(policy.intersection(model.ARM_JOINT_NAMES))

    def test_gains_and_named_scales_match_elf3_and_deployment(self):
        if not controller_audit.DEFAULT_DEPLOYMENT_ROOT.is_dir():
            self.skipTest("User-supplied deployment source is not available on this machine")
        report = controller_audit.build_report()
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["matched_pd_joints"], 29)
        self.assertEqual(report["matched_policy_action_scales"], 15)
        self.assertTrue(report["deployment_adapter_required"])

    def test_control_lookup_rejects_missing_names_and_has_no_nonpolicy_scales(self):
        with self.assertRaises(ValueError):
            model.joint_control_parameters("missing_joint")
        for name in model.DEFAULT_JOINT_POS:
            kp, kd, scale = model.joint_control_parameters(name)
            self.assertGreater(kp, 0)
            self.assertGreater(kd, 0)
            self.assertEqual(scale is not None, bool(re.fullmatch(model.POLICY_JOINT_PATTERN, name)), name)

    def test_preparation_preserves_original_dynamics(self):
        source = ROOT.parent / "data/gl_165_URDF/gl_165.urdf"
        if not source.is_file():
            self.skipTest("Original source asset is not available on this machine")
        before = source.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            output = prepare.prepare(source, Path(temp))
            converted = ET.parse(output).getroot()
            original = ET.fromstring(before)
            for tag in ("joint", "link"):
                for element in converted.findall(tag):
                    other = original.find(f"{tag}[@name='{element.get('name')}']")
                    self.assertIsNotNone(other)
                    # Mesh paths and formatting may change; dynamics must not.
                    for path in ("origin", "axis", "limit", "inertial/origin", "inertial/mass", "inertial/inertia"):
                        if element.find(path) is not None:
                            self.assertEqual(element.find(path).attrib, other.find(path).attrib)
        self.assertEqual(source.read_bytes(), before)

    def test_nominal_pose_places_both_soles_on_ground(self):
        def rotation(axis, angle):
            axis = np.asarray(axis, dtype=float)
            axis /= np.linalg.norm(axis)
            x, y, z = axis
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew

        poses = {"base_link": (np.eye(3), np.array([0, 0, model.STAND_HEIGHT]))}
        for joint in self.root.findall("joint"):
            parent_r, parent_p = poses[joint.find("parent").get("link")]
            origin = joint.find("origin")
            r, p, y = map(float, origin.get("rpy").split())
            offset = np.array(list(map(float, origin.get("xyz").split())))
            joint_r = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
            if joint.get("type") != "fixed":
                joint_r = joint_r @ rotation(list(map(float, joint.find("axis").get("xyz").split())),
                                             model.DEFAULT_JOINT_POS[joint.get("name")])
            poses[joint.find("child").get("link")] = (parent_r @ joint_r, parent_p + parent_r @ offset)
        centres = []
        edges = []
        for name in model.FEET:
            rotation_w, position_w = poses[name]
            contacts = self.root.findall(f"link[@name='{name}']/collision")
            self.assertEqual(len(contacts), 9)
            heights = [(rotation_w @ np.fromstring(c.find('origin').get('xyz'), sep=' ') + position_w)[2]
                       - float(c.find('geometry/sphere').get('radius')) for c in contacts]
            self.assertAlmostEqual(min(heights), 0.0, places=6)
            self.assertGreater(rotation_w[2, 2], 0.9999, name)
            points = np.array([np.fromstring(c.find('origin').get('xyz'), sep=' ') for c in contacts])
            radii = np.array([float(c.find('geometry/sphere').get('radius')) for c in contacts])
            world = points @ rotation_w.T + position_w
            edges.append(((world[:, 1] - radii).min(), (world[:, 1] + radii).max()))
            points[:, 2] -= radii
            centres.append(position_w + rotation_w @ ((points.min(axis=0) + points.max(axis=0)) / 2))
        self.assertAlmostEqual(centres[0][0] - centres[1][0], 0.0, places=6)
        self.assertAlmostEqual(edges[0][0] - edges[1][1], .08, places=6)

    def test_old_stance_uses_its_recorded_arm_asset_signature(self):
        historical = model.load_physics(model.CURRENT_PHYSICS_PROFILE)
        report = ROOT / "legged_lab/envs/gl_165/datasets/arm_disturbance_overhead_261123_2_5ms.json"
        self.assertEqual(model.motion_asset_signature(historical), json.loads(report.read_text())["asset_signature"])
        self.assertNotEqual(historical.default_joint_pos, model.DEFAULT_JOINT_POS)

    def test_xml_armature_and_fixed_head_provenance(self):
        import hashlib
        source = model.PHYSICS_DIR / "source_deployment.xml"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), model.PHYSICS_PARAMETERS['source_xml_sha256'])
        xml = ET.parse(source).getroot()
        self.assertEqual(len(xml.findall('worldbody//joint')), len(model.JOINT_PHYSICS))
        for name, values in model.JOINT_PHYSICS.items():
            joint = xml.find(f".//joint[@name='{values['xml_joint']}']")
            self.assertEqual(float(joint.get('armature')), values['armature_kg_m2'], name)
        for name in model.FIXED_JOINT_POS:
            self.assertEqual(self.root.find(f"joint[@name='{name}']").get('type'), 'fixed')
        self.assertEqual(float(xml.find('option').get('timestep')), .001)
        self.assertEqual(model.PHYSICS_DT, .005)
        self.assertEqual(model.CONTROL_DT, .02)
        self.assertEqual(round(model.CONTROL_DT / model.PHYSICS_DT), 4)
        effective = ET.parse(model.MUJOCO_REFERENCE_PATH).getroot()
        self.assertEqual(float(effective.find('option').get('timestep')), model.PHYSICS_DT)
        self.assertEqual(model.PHYSICS_PARAMETERS['timestep_override']['source_s'], .001)
        self.assertEqual(model.PHYSICS_PARAMETERS['timestep_override']['effective_s'], .005)


class PushTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(enabled=True, curriculum_enabled=True, initial_scale=0.05,
                                   curriculum_duration_s=4800.0,
                                   velocity_ranges={"x": (-.8, .8), "y": (-.8, .8), "yaw": (-.2, .2)})

    def test_tang_velocity_curriculum_uses_simulated_time(self):
        for seconds, scale in [(0, .05), (2400, .525), (4800, 1.), (9000, 1.)]:
            self.assertAlmostEqual(pushes.velocity_push_scale(self.cfg, seconds), scale)

    def test_torso_force_curriculum_endpoints_midpoint_and_plateau(self):
        cfg = SimpleNamespace(initial_force_n=20., final_force_n=100., initial_duration_s=.3,
                              final_duration_s=.8, full_iteration=20000)
        for iteration, force, duration in ((0, 20., .3), (10000, 60., .55),
                                          (20000, 100., .8), (79999, 100., .8)):
            magnitude, length = pushes.torso_force_parameters(cfg, iteration)
            self.assertAlmostEqual(magnitude, force)
            self.assertAlmostEqual(length, duration)

    def test_foot_plant_grace_covers_force_release_and_velocity_kicks(self):
        deadline = torch.zeros(3, dtype=torch.long)
        velocity = torch.tensor([True, False, False])
        torso = SimpleNamespace(started=torch.tensor([False, True, False]), remaining=torch.tensor([0, 300, 0]))
        mask = pushes.update_foot_plant_mask(deadline, 200, 400, velocity, torso)
        self.assertEqual(mask.tolist(), [False, False, True])
        self.assertEqual(deadline.tolist(), [600, 900, 0])
        velocity.zero_()
        torso.started.zero_()
        self.assertEqual(pushes.update_foot_plant_mask(deadline, 600, 400, velocity, torso).tolist(),
                         [True, False, True])
        self.assertTrue(pushes.update_foot_plant_mask(deadline, 900, 400, velocity, torso).all())

    def test_persistent_force_ticks_reset_and_world_direction(self):
        point = torch.tensor([.02, .03, .12])
        body_quat = torch.zeros(3, 2, 4)
        body_quat[:, :, 0] = 1.
        robot = SimpleNamespace(num_instances=3, num_bodies=2, device="cpu",
                                data=SimpleNamespace(com_pos_b=point.expand(3, 2, 3), body_quat_w=body_quat))
        frames = []
        def apply(force_data, torque_data, position_data, indices, is_global):
            self.assertFalse(is_global)
            self.assertIsNone(torque_data)
            torch.testing.assert_close(position_data.view(3, 2, 3)[:, 1], point.expand(3, 3))
            frames.append(force_data.reshape(3, 2, 3).clone())
        robot.root_physx_view = SimpleNamespace(apply_forces_and_torques_at_position=apply)
        pulse = pushes.TorsoForce(robot, 1, .005)
        pulse.start(torch.tensor([0, 2]), 20., .3)
        pulse.start(torch.tensor([1]), 100., .8)
        pulse.reset(torch.tensor([2]))
        target = pulse.force_w[0].clone()
        self.assertAlmostEqual(target.norm().item(), 20., places=4)
        self.assertEqual(target[2].item(), 0.)
        def inverse(quat, force):
            # Independent quaternion-vector rotation, including the inverse sign.
            xyz = -quat[:, 1:]
            t = 2 * torch.cross(xyz, force, dim=-1)
            return force + quat[:, :1] * t + torch.cross(xyz, t, dim=-1)
        with patch.dict(sys.modules, {"isaaclab.utils.math": SimpleNamespace(quat_rotate_inverse=inverse)}):
            for step in range(161):
                if step == 30:
                    robot.data.body_quat_w[:, 1] = torch.tensor([2**-.5, 0., 0., 2**-.5])
                pulse.apply(robot)
        f = torch.stack(frames)
        torch.testing.assert_close(f[:30, 0, 1], target.expand(30, 3))
        rotated = torch.stack((target[1], -target[0], target[2]))
        torch.testing.assert_close(f[30:60, 0, 1], rotated.expand(30, 3), atol=5e-6, rtol=1e-5)
        self.assertEqual(f[60:, 0, 1].count_nonzero().item(), 0)
        torch.testing.assert_close(f[:160, 1, 1].norm(dim=-1), torch.full((160,), 100.))
        self.assertEqual(f[-1].count_nonzero().item(), 0)
        self.assertEqual(f[:, 2].count_nonzero().item() + f[:, :, 0].count_nonzero().item(), 0)

    def test_evaluation_full_strength_and_disabled_baseline(self):
        self.cfg.curriculum_enabled = False
        self.assertEqual(pushes.velocity_push_scale(self.cfg, 0), 1.)
        self.cfg.enabled = False
        self.assertEqual(pushes.velocity_push_scale(self.cfg, 4800), 0.)

    def test_velocity_event_records_only_selected_envs(self):
        velocity = torch.ones(3, 6)
        selected = torch.tensor([0, 2])
        delta = torch.tensor([.2, -.3, 0., 0., 0., .07])
        env = SimpleNamespace(cfg=SimpleNamespace(push=self.cfg), push_scale=lambda: .5,
                              robot=SimpleNamespace(data=SimpleNamespace(root_vel_w=velocity)),
                              push_delta_w=torch.zeros_like(velocity), push_applied=torch.zeros(3, dtype=torch.bool))

        def isaac_event(target, indices, ranges):
            self.assertIs(target, env)
            self.assertEqual(ranges, {"x": (-.4, .4), "y": (-.4, .4), "yaw": (-.1, .1)})
            velocity[indices] += delta

        events = SimpleNamespace(push_by_setting_velocity=isaac_event)
        with patch.dict(sys.modules, {"isaaclab.envs.mdp.events": events}):
            pushes.apply_velocity_push(env, selected)
        torch.testing.assert_close(env.push_delta_w[selected], delta.expand(2, -1))
        self.assertEqual(env.push_delta_w[1].count_nonzero().item(), 0)
        self.assertEqual(env.push_applied.tolist(), [True, False, True])

    def test_repeated_world_xy_velocity_delta_adds_to_current_velocity(self):
        velocity = torch.tensor([[.3, -.2, .5, .4, .2, -.6], [1., 2., 3., 4., 5., 6.]])
        original = velocity.clone()
        ids = torch.tensor([0])
        delta = torch.tensor([[.8, -.7, 0., 0., 0., 0.]])
        def write_root(value, env_ids):
            velocity[env_ids] = value
        env = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(root_vel_w=velocity),
                                                    write_root_velocity_to_sim=write_root),
                              push_delta_w=torch.zeros_like(velocity), push_applied=torch.zeros(2, dtype=torch.bool))
        for _ in range(2):
            pushes.apply_velocity_push(env, ids, delta_w=delta)
        torch.testing.assert_close(velocity[ids], original[ids] + 2 * delta)
        torch.testing.assert_close(velocity[1], original[1])
        torch.testing.assert_close(env.push_delta_w[ids], delta)
        self.assertEqual(env.push_applied.tolist(), [True, False])


class StanceRecoveryTests(unittest.TestCase):
    def test_foot_lock_requires_settled_in_band_frames_after_balance(self):
        off = torch.zeros(3, dtype=torch.bool)
        on = ~off
        gathering, ticks = off, torch.zeros(3, dtype=torch.long)
        def step(recovering, pelvis_lock, in_band, quiet=on):
            nonlocal gathering, ticks
            gathering, ticks, foot_lock = stance.advance_foot_reference(
                gathering, ticks, recovering, pelvis_lock, quiet, in_band, 3)
            return foot_lock.tolist()
        # Only the first robot has already gathered its feet before the pelvis lock.
        one = torch.tensor([True, False, False])
        for _ in range(3):
            self.assertEqual(step(on, off, one), [False] * 3)
        self.assertEqual(step(off, on, one), [True, False, False])
        self.assertEqual(gathering.tolist(), [False, True, True])
        third = torch.tensor([False, False, True])
        self.assertEqual(step(third, off, on), [False] * 3)
        self.assertFalse(gathering[2])
        self.assertEqual(step(off, third, on, ~third), [False] * 3)
        self.assertEqual(step(off, off, on), [False, True, False])
        self.assertEqual(step(off, off, on), [False] * 3)
        self.assertEqual(step(off, off, on), [False, False, True])
        self.assertFalse(gathering.any())

    def test_known_and_unsignalled_pushes_settle_independently(self):
        cfg = stance.DEFAULT_STANCE_RECOVERY
        active = torch.zeros(4, dtype=torch.bool)
        count = torch.zeros(4, dtype=torch.long)
        state = (active, count, count)
        quiet = torch.zeros(4)
        loads = torch.full((4,), 180.)
        def tick(speed, push, right):
            nonlocal state
            *state, relock = stance.advance_stance_reference(
                cfg, .02, *state, push, speed.square(), torch.ones(4), quiet, loads, right)
            return relock
        known = torch.tensor([False, True, False, False])
        tick(torch.tensor([.04, .0, .55, .18]), known, torch.tensor([180.,180.,180.,0.]))
        self.assertEqual(state[0].tolist(), [False, True, True, True])
        for _ in range(100):
            relock = tick(quiet, known, torch.tensor([180.,180.,180.,0.]))
        self.assertEqual(relock.tolist(), [False, False, True, False])
        for _ in range(19):
            self.assertFalse(tick(quiet, active, loads).any())
        self.assertEqual(tick(quiet, active, loads).tolist(), [False, False, False, True])
        for _ in range(79):
            self.assertFalse(tick(quiet, active, loads).any())
        self.assertEqual(tick(quiet, active, loads).tolist(), [False, True, False, False])
        self.assertFalse(tick(quiet, active, loads).any())


class RewardTests(unittest.TestCase):
    def test_standing_gap_and_stagger_count_voluntary_motion(self):
        env = SimpleNamespace(sole_gap_m=torch.tensor([.0281, .0581, .12, .15, .15]),
                              stagger_error_m=torch.tensor([.06, .03, .08, .06, .06]),
                              quiet_gap_active=torch.tensor([True, True, True, True, False]))
        torch.testing.assert_close(-.08 * rewards.quiet_sole_gap(env, .0581, .12, .03),
                                   torch.tensor([-.04, 0., 0., -.04, 0.]))
        torch.testing.assert_close(-.08 * rewards.quiet_foot_stagger(env, .03, .03),
                                   torch.tensor([-.04, 0., -.05, -.04, 0.]))
        ticks = torch.full((5,), 99)
        pushed = torch.tensor([False, False, True, False, False])
        recovering = torch.tensor([False, False, True, True, True])
        prior_push = torch.tensor([False, False, False, True, False])
        valid = torch.tensor([False, True, False, False, False])
        ticks, prior_push, active = stance.advance_gap_timer(ticks, prior_push, pushed, recovering, valid)
        self.assertEqual(ticks.tolist(), [100, 0, 0, 0, 100])
        self.assertEqual(active.tolist(), [True, True, False, False, True])
        ticks, _, _ = stance.advance_gap_timer(ticks, prior_push, pushed & False, recovering & False, valid)
        self.assertEqual(ticks.tolist(), [101, 0, 1, 1, 101])

    def test_quiet_footprint_cost_after_sliding_or_lifting(self):
        xy = torch.zeros(1, 2, 2)
        moved = xy.clone()
        moved[..., 0] += .02
        load = torch.full((1, 2), 100.)
        quiet = torch.tensor([True])
        pushed = ~quiet
        planted = torch.zeros_like(load, dtype=torch.bool)
        anchors, planted = rewards.update_plant_anchors(xy, planted, xy, load, quiet, pushed, 40., 5.)
        reference, valid = rewards.update_quiet_stance_reference(xy, pushed, xy, load, quiet, pushed, 40.)
        anchors, planted = rewards.update_plant_anchors(anchors, planted, moved, torch.tensor([[4., 6.]]), quiet, pushed, 40., 5.)
        self.assertEqual(planted.tolist(), [[False, True]])
        anchors, planted = rewards.update_plant_anchors(anchors, planted, moved, load, quiet, pushed, 40., 5.)
        reference, valid = rewards.update_quiet_stance_reference(reference, valid, moved, load, quiet, pushed, 40.)
        env = SimpleNamespace(quiet_sole_speed=torch.zeros_like(load), quiet_sole_drift=(moved-anchors).norm(dim=-1),
                              quiet_stance_drift=(moved-reference).norm(dim=-1), quiet_stance_valid=valid,
                              quiet_loaded_feet=planted)
        torch.testing.assert_close(rewards.quiet_foot_slide(env, .05, .02, .06), torch.tensor([.75]))
        _, valid = rewards.update_quiet_stance_reference(reference, valid, moved, load, quiet, quiet, 40.)
        self.assertFalse(valid.item())

    def test_standing_target_smoothing_and_fixed_quiet_groups(self):
        previous, next_target = torch.zeros(2, 3), torch.tensor([[0., 0., 0.], [.1, .2, -.3]])
        samples = torch.stack([previous, *arm_motion.arm_substep_targets(previous, next_target, 4)])
        torch.testing.assert_close(samples.diff(dim=0), (next_target / 4).expand_as(samples[1:]))
        env = SimpleNamespace(policy_target_curvature=torch.ones(3), feet_should_be_planted=torch.tensor([True, False, True]),
                              episode_length_buf=torch.tensor([3, 5, 2]))
        torch.testing.assert_close(rewards.quiet_target_curvature(env), torch.tensor([1., 0., 0.]))
        env.no_push_env_count = 2
        env.cfg = SimpleNamespace(push=SimpleNamespace(mode=pushes.MIXED_PUSH))
        with patch.object(pushes, "apply_mixed_push") as event:
            pushes.apply_push_outside_quiet_envs(env, torch.tensor([1, 0]))
            event.assert_not_called()
            pushes.apply_push_outside_quiet_envs(env, torch.tensor([3, 1, 2]))
            self.assertEqual(event.call_args.args[1].tolist(), [3, 2])

    def test_fore_aft_offset_uses_sole_centres_in_the_pelvis_heading(self):
        heading = torch.tensor([0., math.pi / 2, math.pi, 0., 0.])
        zero = torch.zeros_like(heading)

        def quat(yaw):
            return torch.stack(((yaw / 2).cos(), torch.zeros_like(yaw),
                                torch.zeros_like(yaw), (yaw / 2).sin()), dim=-1)

        forward = torch.stack((heading.cos(), heading.sin(), zero), dim=-1)
        lateral = torch.stack((-heading.sin(), heading.cos(), zero), dim=-1)
        right = torch.tensor([[1., -2., .1]]).expand(len(heading), -1)
        left = right + .24 * lateral + torch.tensor([.1, -.1, .1, 0., 0.])[:, None] * forward
        left[:, 2] += .1
        foot_yaws = heading[:, None].expand(-1, 2).clone()
        foot_yaws[-1, 1] = math.pi / 2
        data = SimpleNamespace(body_pos_w=torch.stack((left, right), dim=1),
                               body_quat_w=quat(foot_yaws), root_quat_w=quat(heading))
        env = SimpleNamespace(robot=SimpleNamespace(data=data), feet_body_ids=[0, 1],
                              sole_centers_b=torch.tensor([[.08, 0., -.04], [.08, 0., -.04]]))
        torch.testing.assert_close(rewards.sole_fore_aft_offset(env), torch.tensor([.1, .1, .1, 0., .08]))

    def test_post_push_stagger_persists_when_still_and_has_a_free_band(self):
        env = SimpleNamespace(fore_aft_offset_m=torch.tensor([.2, 0., .02, .03, .05, .08, .15]),
                              post_push_settled=torch.tensor([False, True, True, True, True, True, True]))
        cost = rewards.post_push_feet_stagger(env, .03, .05)
        torch.testing.assert_close(cost, torch.tensor([0., 0., 0., 0., .08, .5, 1.9]))
        # Fifty identical 20 ms frames cost one full second, even with stationary soles.
        second = sum(-.02 * rewards.post_push_feet_stagger(env, .03, .05) for _ in range(50))
        torch.testing.assert_close(second, -cost)

    def test_loaded_contact_speed_uses_material_points_and_current_ground_load(self):
        link = torch.zeros(7, 13)
        link[:, :3] = torch.tensor([4., -2., .05])
        link[:3, 7:9] = torch.tensor([.03, -.04])
        link[3, 7] = .05
        link[3, 11] = 1.0  # rocking about a stationary contact .05 m below the link
        link[4:6, 12] = 1.0
        forces = torch.tensor([40., 0., 39., 0., 0., 0., 80., 0., 20., 20., 10., 30., 0., 0.])[:, None]
        points = link[:, None, :3].expand(-1, 2, -1).clone()
        points[..., 2] = 0.
        points[4, :, 0] += torch.tensor([-.1, .1])  # opposite slip velocities must not cancel
        points[5, 1, 0] += .2  # load-weighted mean: (10*0 + 30*.2) / 40
        normals = torch.zeros(14, 3)
        normals[:, 2] = 1.
        starts = torch.tensor([[0], [2], [-1], [6], [8], [10], [250]], dtype=torch.int32)
        counts = torch.tensor([[1], [1], [0], [1], [2], [2], [0]], dtype=torch.int32)
        data = (forces, points.reshape(-1, 3), normals, None, counts, starts)
        speeds = rewards.loaded_contact_speed(link, data, torch.arange(2), 40.)
        torch.testing.assert_close(speeds, torch.tensor([.05, 0., 0., 0., .1, .15, 0.]))

    def test_loaded_slide_charges_each_foot_once_including_recovery(self):
        # Metres travelled in four 5 ms substeps. Only the loaded feet contributed.
        distance = torch.tensor([[.002, .002], [.002, 0.], [0., .001], [0., 0.]])
        env = SimpleNamespace(sole_slide_distance=distance, step_dt=.020,
                              recovering=torch.tensor([True, True, False, False]))
        reward = -.5 * env.step_dt * rewards.loaded_sole_slide(env)
        torch.testing.assert_close(reward, torch.tensor([-.002, -.001, -.0005, 0.]))

    def test_quiet_inward_slip_uses_loaded_sole_surface_velocity(self):
        # Inward, outward, common translation, recovery, unloaded, wide, rotated heading,
        # rolling about a stationary sole, and the middle of the width transition.
        count = 9
        feet = torch.zeros(count, 2, 13)
        feet[..., 3] = 1.
        feet[:, 0, 8], feet[:, 1, 8] = -.005, .005
        feet[1, :, 8] *= -1
        feet[2, :, 8] = .2
        feet[6, :, 8] = 0.
        feet[6, 0, 7], feet[6, 1, 7] = .005, -.005
        feet[7, :, 10] = torch.tensor([1., -1.])
        feet[7, :, 8] = torch.tensor([-.04, .04])
        root = feet[:, 0, 3:7].clone()
        root[6] = torch.tensor([2**-.5, 0., 0., 2**-.5])
        spheres = torch.zeros(2, 9, 4)
        spheres[..., 2], spheres[..., 3] = -.035, .005
        gap = torch.full((count,), .0581)
        gap[5], gap[8] = .08, .06215
        quiet = torch.ones(count, dtype=torch.bool)
        quiet[3] = False
        loads = torch.full((count, 2, 3), 190.)
        loads[4, 1, 2] = 20.
        env = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(
            body_link_state_w=feet, root_quat_w=root)), sole_spheres=spheres, sole_gap_m=gap,
            feet_body_ids=[0, 1], feet_cfg=SimpleNamespace(body_ids=[0, 1]),
            contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w=loads)),
            feet_should_be_planted=quiet)
        cost = rewards.quiet_inward_slip(env, .0581, .0081, .01, 40., .001)
        torch.testing.assert_close(cost, torch.tensor([.5, 0., 0., 0., 0., 0., .5, 0., .25]))

    def test_all_phase_and_historical_upper_gap_cost(self):
        env = SimpleNamespace(sole_gap_m=torch.tensor([.50, .02, .12, .17, .32, .52]),
                              recovering=torch.tensor([True, False, False, False, False, False]))
        torch.testing.assert_close(rewards.feet_width_excess(env, .12, .10),
                                   torch.tensor([0., 0., 0., .125, 1.5, 3.5]))
        torch.testing.assert_close(rewards.feet_width_excess(env, .12, .10, all_phases=True),
                                   torch.tensor([3.3, 0., 0., .125, 1.5, 3.5]))
        self.assertGreater(rewards.feet_clearance_shortfall(env, .03)[1].item(), 0.)

    def test_quiet_foot_lift_uses_lowest_sphere_and_per_foot_cost(self):
        contacts = model.PHYSICS_PARAMETERS["foot_contacts"]
        spheres = torch.tensor([[[*p["position_m"], p["radius_m"]] for p in contacts[foot]] for foot in model.FEET])
        quat = torch.zeros(3, 2, 4)
        quat[..., 0] = 1.
        quat[2, 0] = torch.tensor([math.cos(.2), 0., math.sin(.2), 0.])
        feet = torch.zeros(3, 2, 3)
        ground = torch.tensor([[0., 0., .7], [0., 0., 0.], [0., 0., 0.]])
        feet[..., 2] = ground[:, 2:3] + .04
        feet[1, :, 2] += torch.tensor([.015, .035])
        # A pitched foot can still rest on its forward sphere.
        feet[2, 0, 2] = .145 * math.sin(.4) + .035 * math.cos(.4) + .005
        env = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=feet, body_quat_w=quat)),
                              feet_body_ids=[0, 1], sole_spheres=spheres, scene=SimpleNamespace(env_origins=ground))
        env.sole_height_m, env.sole_high_m = rewards.sole_height_bounds(env)
        torch.testing.assert_close(env.sole_height_m, torch.tensor([[0., 0.], [.015, .035], [0., 0.]]))
        env.feet_should_be_planted = torch.ones(3, dtype=torch.bool)
        torch.testing.assert_close(rewards.quiet_feet_lift(env, .005, .02, .015, .25),
                                   torch.tensor([0., 1.5, .25]))
        env.feet_should_be_planted[1:] = False
        self.assertEqual(rewards.quiet_feet_lift(env, .005, .02, .015, .25).count_nonzero().item(), 0)

    def test_double_support_bonus_is_neutral_during_recovery(self):
        env = SimpleNamespace(feet_cfg=SimpleNamespace(body_ids=[0, 1]), contact_sensor=SimpleNamespace(
            data=SimpleNamespace(net_forces_w=torch.zeros(3, 2, 3))))
        env.contact_sensor.data.net_forces_w[:, :, 2] = torch.tensor([[180., 185.], [180., 185.], [0., 260.]])
        torch.testing.assert_close(rewards.both_feet_loaded(env), torch.tensor([1., 1., 0.]))
        env.feet_should_be_planted = torch.tensor([True, False, False])
        torch.testing.assert_close(rewards.both_feet_loaded(env), torch.tensor([1., 0., 0.]))

    def test_foot_load_shortfall_penalizes_each_foot_only_while_planted(self):
        loads = torch.tensor([[212., 212.], [150., 150.], [149., 300.], [305.5, 118.8],
                              [118.8, 305.5], [75., 75.], [0., 300.], [-2., 0.], [0., 20.]])
        forces = torch.full((len(loads), 4, 3), 500.)
        forces[:, [3, 1], 2] = loads
        env = SimpleNamespace(contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w=forces)),
                              feet_cfg=SimpleNamespace(body_ids=[3, 1]),
                              feet_should_be_planted=torch.tensor([True] * 8 + [False]))
        expected = torch.tensor([0., 0., 1/150, 31.2/150, 31.2/150, 1., 1., 2., 0.])
        torch.testing.assert_close(rewards.foot_load_shortfall(env, 150.), expected)
        # A continuing load deficit incurs the same cost on every 20 ms step.
        second = sum(-.02 * rewards.foot_load_shortfall(env, 150.) for _ in range(50))
        torch.testing.assert_close(second, -expected)

    def test_sole_clearance_follows_pelvis_and_foot_orientation(self):
        # Same stance in two world headings, a narrow stance and an inward-turned left toe.
        heading = torch.tensor([0., math.pi / 2, 0., 0.])
        foot_yaw = heading[:, None].expand(-1, 2).clone()
        foot_yaw[-1, 0] -= math.pi / 2

        def quat(yaw):
            zero = torch.zeros_like(yaw)
            return torch.stack(((yaw / 2).cos(), zero, zero, (yaw / 2).sin()), dim=-1)

        side = torch.stack((-heading.sin(), heading.cos(), torch.zeros_like(heading)), dim=-1)
        width = torch.tensor([.12, .12, .10, .14])
        right = torch.zeros(4, 3)
        left = width[:, None] * side
        left[0, 0] = .2
        left[0, 2] = .15
        contacts = model.PHYSICS_PARAMETERS["foot_contacts"]
        env = SimpleNamespace(feet_body_ids=[0, 1], sole_spheres=torch.tensor([
            [[*p["position_m"], p["radius_m"]] for p in contacts[foot]] for foot in model.FEET
        ]), robot=SimpleNamespace(data=SimpleNamespace(root_quat_w=quat(heading),
            body_quat_w=quat(foot_yaw), body_pos_w=torch.stack((left, right), dim=1))))
        env.sole_gap_m = rewards.sole_inner_gap(env)
        torch.testing.assert_close(env.sole_gap_m, torch.tensor([.035, .035, .015, -.0525]))
        subset = torch.tensor([3, 0])
        torch.testing.assert_close(rewards.sole_inner_gap(env, subset), env.sole_gap_m[subset])
        torch.testing.assert_close(rewards.feet_clearance_shortfall(env, .03),
                                   torch.tensor([0., 0., .25, 2.75**2]))
        env.sole_gap_m = torch.tensor([.03, .06, .08, .12])
        torch.testing.assert_close(rewards.feet_clearance_shortfall(env, .08),
                                   torch.tensor([(.05 / .08)**2, .0625, 0., 0.]))
        env.sole_gap_m = torch.tensor([.03, .05, .058131])
        torch.testing.assert_close(rewards.feet_clearance_shortfall(env, .05), torch.tensor([.16, 0., 0.]))

    def test_waist_zero_uses_only_actual_waist_angles_and_joint_zero(self):
        position = torch.tensor([[2., 0., 3., 0., 0.], [2., .1, 3., -.2, .3]])
        env = SimpleNamespace(waist_joint_ids=[4, 1, 3], robot=SimpleNamespace(data=SimpleNamespace(
            joint_pos=position, default_joint_pos=torch.ones_like(position))))
        torch.testing.assert_close(rewards.waist_zero(env), torch.tensor([0., .14]))
        position[:, [0, 2]] = 50.0
        torch.testing.assert_close(rewards.waist_zero(env), torch.tensor([0., .14]))

    def test_exogenous_arm_motion_is_not_penalized_as_policy_motion(self):
        position = torch.tensor([[0., 0., 1.0], [0.1, 0.2, 1.5]])
        env = SimpleNamespace(policy_joint_ids=[0, 1], effort_limits=torch.ones_like(position),
                              robot=SimpleNamespace(data=SimpleNamespace(joint_pos=position,
                                  default_joint_pos=torch.zeros_like(position), applied_torque=position)))
        torch.testing.assert_close(rewards.nominal_pose(env), torch.tensor([0., 0.05]))
        torch.testing.assert_close(rewards.normalized_effort(env), torch.tensor([0., 0.05]))

    def test_wxyz_heading_and_wraparound(self):
        angles = torch.tensor([0.0, math.pi / 2, math.pi - 0.01])
        quat = torch.stack([torch.cos(angles / 2), torch.zeros(3), torch.zeros(3), torch.sin(angles / 2)], dim=-1)
        torch.testing.assert_close(rewards.yaw_from_quat(quat), angles)
        env = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(root_quat_w=quat)),
                              stand_anchor_yaw=torch.tensor([0.0, 0.0, -math.pi + 0.01]))
        self.assertAlmostEqual(rewards.heading_error(env)[2].item(), -0.02, places=5)

    def test_torso_sway_measures_chest_tilt_and_ramps_after_external_pushes(self):
        q = torch.zeros(6, 2, 4)
        q[:, :, 0] = 1.
        q[1:3, 1] = torch.tensor([math.sqrt(.5), 0., math.sqrt(.5), 0.])
        omega = torch.zeros(6, 2, 3)
        omega[:, 0, 1] = 40.  # Rapid pelvis motion is independent of this reward.
        omega[:, 1, 0] = .5
        omega[2, 1] = torch.tensor([0., 0., .5])
        env = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(body_quat_w=q, body_ang_vel_w=omega)),
                              torso_body_id=1, sim_step_counter=450, physics_dt=.005,
                              foot_recovery_until=torch.tensor([0, 0, 0, 500, 350, 250]))
        cost = rewards.torso_sway(env, rate_scale_rad_s=.5, ramp_s=1.)
        torch.testing.assert_close(cost, torch.tensor([.5, 0., .5, 0., .25, .5]))
        env.foot_recovery_until[4] = 0  # An episode reset restores full quiet-stance weighting.
        self.assertAlmostEqual(rewards.torso_sway(env, .5, 1.)[4].item(), .5)

    def test_upside_down_torso_cannot_earn_upright_reward(self):
        quats = torch.tensor([[[1., 0., 0., 0.]], [[0., 1., 0., 0.]]])
        env = SimpleNamespace(torso_body_id=0, robot=SimpleNamespace(data=SimpleNamespace(body_quat_w=quats)))
        result = rewards.torso_upright(env)
        self.assertEqual(result[0].item(), 1.0)
        self.assertLess(result[1].item(), 1e-10)

    def test_foot_heading_is_relative_to_pelvis_and_wraps_at_pi(self):
        def quats(angles):
            zeros = torch.zeros_like(angles)
            return torch.stack((torch.cos(angles / 2), zeros, zeros, torch.sin(angles / 2)), dim=-1)

        base = torch.tensor([0., 1.2, math.pi - .01])
        offsets = torch.tensor([[0., 0.], [-.2, .2], [.03, -.03]])
        env = SimpleNamespace(feet_body_ids=[0, 1], robot=SimpleNamespace(data=SimpleNamespace(
            root_quat_w=quats(base), body_quat_w=quats(base[:, None] + offsets))))
        torch.testing.assert_close(rewards.feet_heading_error(env), offsets)
        expected = torch.tensor([0., 2 * (.2 - math.radians(3))**2, 0.])
        torch.testing.assert_close(rewards.feet_heading(env, math.radians(3)), expected)
        # Rotating the whole robot must not change the foot-heading penalty.
        env.robot.data.root_quat_w = quats(base + 1.7)
        env.robot.data.body_quat_w = quats(base[:, None] + offsets + 1.7)
        torch.testing.assert_close(rewards.feet_heading(env, math.radians(3)), expected)


class ArmMotionTests(unittest.TestCase):
    def setUp(self):
        import json
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "arm.npz"
        self.names = ["left", "right"]
        self.nominal = torch.zeros(4, 2)
        values = np.array([[[0, 0], [.1, -.1], [0, 0]], [[0, 0], [.2, -.2], [0, 0]]], dtype=np.float32)
        np.savez(self.path, targets=values, joint_names=self.names, control_dt=.02,
                 metadata=np.asarray(json.dumps({"asset_signature": "asset-v1"})))

    def tearDown(self):
        self.temp.cleanup()

    def load(self, signature="asset-v1", step_dt=.02):
        return arm_motion.ArmMotionDatabase(self.path, self.names, self.nominal, signature, 4, "cpu", step_dt)

    def test_clip_switches_and_partial_resets_are_continuous(self):
        database = self.load()
        first = database.step(.5).clone()
        self.assertTrue((first[:, 0] > 0).all())
        self.assertTrue((first[:, 0] <= .1).all())
        database.reset(torch.tensor([0]))
        torch.testing.assert_close(database.target[0], self.nominal[0])
        torch.testing.assert_close(database.target[1:], first[1:])
        database.step(1)
        torch.testing.assert_close(database.target[1:], self.nominal[1:])
        database.step(1)
        torch.testing.assert_close(database.target, self.nominal)
        torch.testing.assert_close(database.step(0), self.nominal)

    def test_stale_asset_and_wrong_timestep_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            self.load(signature="changed-geometry")
        with self.assertRaisesRegex(ValueError, "sampling rate"):
            self.load(step_dt=.01)


if __name__ == "__main__":
    unittest.main()
