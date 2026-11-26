"""Compare standing and sole width after one recorded disturbance per environment."""

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--reference", type=Path, help="Replay the per-environment disturbances from a previous report.")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
from legged_lab.assets.gl_165.disturbances import MIXED_PUSH, VELOCITY_ONLY
from legged_lab.envs.gl_165.pushes import apply_mixed_push, apply_velocity_push
from legged_lab.envs.gl_165.rewards import sole_inner_gap
from legged_lab.envs.gl_165.stand_cfg import Gl165StandAgentCfg, Gl165StandEnvCfg
from legged_lab.envs.gl_165.stand_env import Gl165StandEnv


@torch.inference_mode()
def evaluate():
    cfg = Gl165StandEnvCfg()
    cfg.restore_for_playback(args.checkpoint)
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and "world_velocity_deltas" in reference:
        cfg.restore_push_profile({"push_mode": VELOCITY_ONLY, "velocity_push": reference["velocity_push"]})
    force_test = cfg.push.mode == MIXED_PUSH
    if reference and force_test != ("world_forces_xy_n" in reference):
        raise ValueError("The checkpoint and reference require different disturbance modes")
    cfg.device, cfg.scene.num_envs, cfg.scene.seed = args.device, 128, 1121
    cfg.noise.add_noise = False
    cfg.arm_motion.curriculum_enabled = cfg.push.curriculum_enabled = False
    velocity_profile = cfg.push.velocity_profile()
    onset_s = 2.0 if force_test else 8.0
    cfg.push.interval_s = (onset_s, onset_s)
    if force_test:
        cfg.push.torso_force.event_probability = 1.0
    env = Gl165StandEnv(cfg, headless=True)
    # The comparison calls for one full push and moving arms in every test case.
    env.no_push_env_count = 0
    agent = Gl165StandAgentCfg()
    agent.device = env.device
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=env.device)
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    n, dt = env.num_envs, env.step_dt
    duration_s = 0.0
    counts = torch.zeros(n, dtype=torch.long, device=env.device)
    first_tick = torch.full_like(counts, -1)
    actual = torch.zeros(n, 2 if force_test else 6, device=env.device)
    if force_test:
        env.learning_iteration = cfg.push.torso_force.full_iteration
        force_n, duration_s = cfg.push.torso_force.final_force_n, cfg.push.torso_force.final_duration_s
        if reference:
            preset = torch.tensor(reference["world_forces_xy_n"], device=env.device)
            if reference["duration_s"] != duration_s or reference["force_n"] != force_n:
                raise ValueError("The comparison requires the same torso force and duration")
            assert preset.shape == actual.shape and torch.allclose(preset.norm(dim=1), torch.full_like(preset[:, 0], force_n))
    else:
        axes = ("x", "y", "z", "roll", "pitch", "yaw")
        lo, hi = torch.tensor([cfg.push.velocity_ranges.get(axis, (0., 0.)) for axis in axes]).T
        if reference:
            preset = torch.tensor(reference["world_velocity_deltas"])
        else:
            rng = torch.Generator().manual_seed(cfg.scene.seed)
            preset = lo + (hi - lo) * torch.rand(n, 6, generator=rng)
        assert preset.shape == actual.shape and ((lo - 1.e-6 <= preset) & (preset <= hi + 1.e-6)).all()
        preset = preset.to(env.device)
    force_ticks = round(duration_s / dt)

    def schedule(task, env_ids):
        if force_test:
            apply_mixed_push(task, env_ids)
            if reference:
                task.torso_force.force_w[env_ids, :2] = preset[env_ids]
            actual[env_ids] = task.torso_force.force_w[env_ids, :2]
        else:
            apply_velocity_push(task, env_ids, delta_w=preset[env_ids])
            actual[env_ids] = task.push_delta_w[env_ids]
        first_tick[env_ids] = tick
        counts[env_ids] += 1

    term = env.event_manager.get_term_cfg("push_robot")
    term.func, term.interval_range_s = schedule, (60., 60.)
    obs, _ = env.get_observations()
    failed = torch.zeros_like(counts, dtype=torch.bool)
    timed_out = torch.zeros_like(failed)
    relocked = torch.zeros_like(failed)
    elapsed = (3, 5)
    gaps = {s: torch.full((n,), float("nan"), device=env.device) for s in elapsed}
    reached = {s: torch.zeros_like(failed) for s in elapsed}
    balanced = {s: torch.zeros_like(failed) for s in elapsed}
    simulated_s = 9. if force_test else onset_s + 6.
    for tick in range(1, round(simulated_s / dt) + 1):
        obs, reward, done, extras = env.step(policy(obs))
        if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
            raise ValueError("Nonfinite policy evaluation")
        failed |= done & ~extras["time_outs"]
        timed_out |= done & extras["time_outs"]
        since_release = tick - first_tick - force_ticks
        eligible = (counts == 1) & ~failed & ~timed_out
        relocked |= eligible & (since_release >= 0) & env.stance_relocked
        measured_gap = env.sole_gap_m if hasattr(env, "sole_gap_m") else sole_inner_gap(env)
        for s in elapsed:
            at_sample = (counts == 1) & (since_release == round(s / dt))
            reached[s] |= at_sample
            valid = at_sample & eligible
            gaps[s][valid] = measured_gap[valid]
            balanced[s] |= valid & ~env.recovering
    assert (counts == 1).all() and all(v.all() for v in reached.values())
    if force_test:
        assert torch.allclose(actual.norm(dim=1), torch.full_like(actual[:, 0], force_n))
    else:
        torch.testing.assert_close(actual, preset)
    if reference:
        assert first_tick.tolist() == reference["force_start_control_ticks" if force_test else "push_start_control_ticks"]
        torch.testing.assert_close(actual, preset)
    event_report = (dict(force_n=force_n, duration_s=duration_s, force_start_control_ticks=first_tick.tolist(),
                         world_forces_xy_n=actual.tolist()) if force_test else
                    dict(velocity_push=velocity_profile, push_start_control_ticks=first_tick.tolist(),
                         world_velocity_deltas=preset.tolist()))
    cuts = {}
    for s in elapsed:
        valid = torch.isfinite(gaps[s])
        values = gaps[s][valid]
        in_band = valid & (gaps[s] >= .03) & (gaps[s] <= .12)
        cuts[str(s)] = dict(survivors=int(valid.sum()), failed_or_timed_out=n-int(valid.sum()),
                            balanced_with_3_to_12cm=int((in_band & balanced[s]).sum()),
                            below_training_minimum=int((values < env.min_foot_clearance_m).sum()) if env.min_foot_clearance_m else None,
                            median_m=float(values.quantile(.5)) if len(values) else None,
                            p90_m=float(values.quantile(.9)) if len(values) else None)
        if env.max_foot_clearance_m is not None:
            settled_gaps = gaps[s][valid & balanced[s]]
            inside = (settled_gaps >= env.min_reference_gap_m) & (settled_gaps <= env.max_foot_clearance_m)
            cuts[str(s)]["balanced_within_training_band"] = int(inside.sum())
            cuts[str(s)]["balanced_above_training_maximum"] = int((settled_gaps > env.max_foot_clearance_m).sum())
    result = dict(checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  seed=cfg.scene.seed, environments=n, push_mode=cfg.push.mode, **event_report, observation_noise=False,
                  simulated_seconds=simulated_s, full_arm_motion=True, control_dt_s=dt,
                  min_foot_clearance_m=env.min_foot_clearance_m, max_foot_clearance_m=env.max_foot_clearance_m,
                  quiet_band_minimum_m=env.min_reference_gap_m,
                  failures=int(failed.sum()), timeouts=int(timed_out.sum()),
                  survived_and_relocked=int((relocked & ~env.recovering & ~failed & ~timed_out).sum()),
                  gap_after_release_s=cuts,
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest() if reference else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(args.output)
    print(json.dumps({"standing":result["survived_and_relocked"], "gap_after_release_s":cuts}), flush=True)
    env.sim.clear_all_callbacks()
    env.sim.clear_instance()


if __name__ == "__main__":
    try:
        evaluate()
    finally:
        app.close(wait_for_replicator=False)
