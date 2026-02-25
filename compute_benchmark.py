"""
Compute Isaac Lab reward-style benchmark metrics from sim-eval state logs.

Computes per-episode and aggregate metrics that mirror the reward functions in
isaaclab.envs.mdp.rewards.common, enabling quantitative comparison across
quantization configurations (e.g. bf16 baseline vs fp8, int8, etc.).

Usage:
    python compute_benchmark.py runs/2026-02-24/22-12-54/state_logs/
    python compute_benchmark.py runs/2026-02-24/22-12-54/state_logs/ --compare runs/2026-02-25/10-00-00/state_logs/
"""

import argparse
import json
import numpy as np
from pathlib import Path


def compute_episode_metrics(episode_data: dict) -> dict:
    """Compute benchmark metrics for a single episode."""
    states = episode_data["states"]
    if len(states) == 0:
        return {}

    defaults = np.array(episode_data["arm_default_joint_pos"])
    pos_limits = np.array(episode_data["arm_soft_joint_pos_limits"])  # [7, 2]
    vel_limits = np.array(episode_data["arm_soft_joint_vel_limits"])  # [7]

    positions = np.array([s["arm_joint_pos"] for s in states])
    velocities = np.array([s["arm_joint_vel"] for s in states])
    accelerations = np.array([s["arm_joint_acc"] for s in states])
    applied_torques = np.array([s["arm_applied_torque"] for s in states])
    computed_torques = np.array([s["arm_computed_torque"] for s in states])
    actions = np.array([s["action"] for s in states])

    metrics = {}

    # --- Joint penalties (per-step mean of sum-over-joints) ---

    # joint_torques_l2: control effort
    metrics["joint_torques_l2"] = float(np.mean(np.sum(applied_torques ** 2, axis=1)))

    # joint_vel_l1: velocity magnitude (L1)
    metrics["joint_vel_l1"] = float(np.mean(np.sum(np.abs(velocities), axis=1)))

    # joint_vel_l2: velocity magnitude (L2 squared)
    metrics["joint_vel_l2"] = float(np.mean(np.sum(velocities ** 2, axis=1)))

    # joint_acc_l2: jerkiness indicator
    metrics["joint_acc_l2"] = float(np.mean(np.sum(accelerations ** 2, axis=1)))

    # joint_deviation_l1: deviation from default pose
    metrics["joint_deviation_l1"] = float(np.mean(np.sum(np.abs(positions - defaults), axis=1)))

    # joint_pos_limits: soft position limit violations
    lower_violation = np.clip(-(positions - pos_limits[:, 0]), a_min=0, a_max=None)
    upper_violation = np.clip(positions - pos_limits[:, 1], a_min=0, a_max=None)
    metrics["joint_pos_limits"] = float(np.mean(np.sum(lower_violation + upper_violation, axis=1)))

    # joint_vel_limits: velocity limit violations (soft_ratio=1.0, clipped to 1.0)
    vel_violation = np.clip(np.abs(velocities) - vel_limits, a_min=0, a_max=1.0)
    metrics["joint_vel_limits"] = float(np.mean(np.sum(vel_violation, axis=1)))

    # applied_torque_limits: torque saturation (gap between desired and applied)
    metrics["applied_torque_limits"] = float(np.mean(np.sum(np.abs(applied_torques - computed_torques), axis=1)))

    # --- Action penalties ---

    # action_l2: action magnitude
    metrics["action_l2"] = float(np.mean(np.sum(actions ** 2, axis=1)))

    # action_rate_l2: action smoothness (rate of change)
    if len(actions) > 1:
        action_diffs = actions[1:] - actions[:-1]
        metrics["action_rate_l2"] = float(np.mean(np.sum(action_diffs ** 2, axis=1)))
    else:
        metrics["action_rate_l2"] = 0.0

    # --- End-effector metrics ---

    if "ee_pose" in states[0]:
        ee_poses = np.array([s["ee_pose"] for s in states])  # [T, 7] (x,y,z,qw,qx,qy,qz)
        ee_positions = ee_poses[:, :3]

        # EE path length (total distance traveled)
        ee_diffs = np.diff(ee_positions, axis=0)
        metrics["ee_path_length"] = float(np.sum(np.linalg.norm(ee_diffs, axis=1)))

        # EE smoothness: L2 of position acceleration (second derivative)
        if len(ee_positions) > 2:
            ee_acc = np.diff(ee_positions, n=2, axis=0)
            metrics["ee_acc_l2"] = float(np.mean(np.sum(ee_acc ** 2, axis=1)))
        else:
            metrics["ee_acc_l2"] = 0.0

    if "ee_vel" in states[0]:
        ee_vels = np.array([s["ee_vel"] for s in states])  # [T, 6] (vx,vy,vz,wx,wy,wz)
        ee_lin_vel = ee_vels[:, :3]

        # EE linear velocity magnitude (mean speed)
        metrics["ee_speed_mean"] = float(np.mean(np.linalg.norm(ee_lin_vel, axis=1)))

    # --- Object tracking metrics ---

    has_target = "object_positions" in states[0] and states[0]["object_positions"].get("target")
    has_container = "object_positions" in states[0] and states[0]["object_positions"].get("container")
    if has_target and has_container:
        target_positions = np.array([s["object_positions"]["target"] for s in states])
        container_positions = np.array([s["object_positions"]["container"] for s in states])

        # Distance between target and container over time
        obj_dists = np.linalg.norm(target_positions - container_positions, axis=1)
        metrics["obj_dist_final"] = float(obj_dists[-1])
        metrics["obj_dist_min"] = float(np.min(obj_dists))

        # EE-to-target distance (how close gripper got to the object)
        if "ee_pose" in states[0]:
            ee_to_target = np.linalg.norm(ee_positions - target_positions, axis=1)
            metrics["ee_target_dist_min"] = float(np.min(ee_to_target))

    # --- Inference timing ---

    if "inference_time_ms" in states[0]:
        infer_times = np.array([s["inference_time_ms"] for s in states])
        metrics["inference_time_ms_mean"] = float(np.mean(infer_times))
        metrics["inference_time_ms_p95"] = float(np.percentile(infer_times, 95))

        # Model-call-only timing (excludes cached action replays)
        if "model_call" in states[0]:
            model_call_times = np.array([s["inference_time_ms"] for s in states if s["model_call"]])
            if len(model_call_times) > 0:
                metrics["model_inference_ms_mean"] = float(np.mean(model_call_times))
                metrics["model_inference_ms_p95"] = float(np.percentile(model_call_times, 95))

    # --- Summary stats ---
    metrics["success"] = episode_data.get("success", False)
    metrics["num_steps"] = episode_data.get("num_steps", len(states))

    return metrics


def load_run(state_logs_dir: Path) -> list[dict]:
    """Load all episode state logs from a directory."""
    episodes = []
    for f in sorted(state_logs_dir.glob("episode_*_state.json")):
        with open(f) as fh:
            episodes.append(json.load(fh))
    return episodes


def aggregate_metrics(episode_metrics: list[dict]) -> dict:
    """Compute mean and std across episodes."""
    if not episode_metrics:
        return {}

    keys = [k for k in episode_metrics[0] if k not in ("success", "num_steps")]
    agg = {}
    for k in keys:
        values = [m[k] for m in episode_metrics]
        agg[k] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    successes = [m["success"] for m in episode_metrics]
    agg["success_rate"] = float(np.mean(successes))
    agg["num_episodes"] = len(episode_metrics)
    agg["mean_steps"] = float(np.mean([m["num_steps"] for m in episode_metrics]))

    return agg


def print_metrics(name: str, agg: dict):
    """Pretty-print aggregated metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Episodes: {agg['num_episodes']}, Success rate: {agg['success_rate']:.1%}, Mean steps: {agg['mean_steps']:.0f}")
    print(f"{'─' * 60}")
    print(f"  {'Metric':<28} {'Mean':>12} {'Std':>12}")
    print(f"{'─' * 60}")
    for k in sorted(agg):
        if isinstance(agg[k], dict):
            print(f"  {k:<28} {agg[k]['mean']:>12.4f} {agg[k]['std']:>12.4f}")
    print(f"{'=' * 60}")


def print_comparison(name_a: str, agg_a: dict, name_b: str, agg_b: dict):
    """Print side-by-side comparison of two runs."""
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON: {name_a} vs {name_b}")
    print(f"{'=' * 80}")
    print(f"  {'':>28} {name_a:>16} {name_b:>16} {'Delta':>12}")
    print(f"{'─' * 80}")
    print(f"  {'success_rate':<28} {agg_a['success_rate']:>15.1%} {agg_b['success_rate']:>15.1%} {agg_b['success_rate'] - agg_a['success_rate']:>+11.1%}")

    keys = sorted(k for k in agg_a if isinstance(agg_a.get(k), dict) and isinstance(agg_b.get(k), dict))
    for k in keys:
        ma, mb = agg_a[k]["mean"], agg_b[k]["mean"]
        delta_pct = ((mb - ma) / ma * 100) if ma != 0 else float("inf")
        print(f"  {k:<28} {ma:>16.4f} {mb:>16.4f} {delta_pct:>+11.1f}%")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Compute benchmark metrics from sim-eval state logs")
    parser.add_argument("state_logs_dir", type=Path, help="Path to state_logs/ directory")
    parser.add_argument("--compare", type=Path, default=None, help="Second state_logs/ directory to compare against")
    parser.add_argument("--output", type=Path, default=None, help="Save metrics JSON to this path")
    args = parser.parse_args()

    # Compute metrics for primary run
    episodes = load_run(args.state_logs_dir)
    if not episodes:
        print(f"No episode files found in {args.state_logs_dir}")
        return

    episode_metrics = [compute_episode_metrics(ep) for ep in episodes]
    agg = aggregate_metrics(episode_metrics)
    print_metrics(str(args.state_logs_dir), agg)

    # Optionally compare with second run
    if args.compare:
        episodes_b = load_run(args.compare)
        if not episodes_b:
            print(f"No episode files found in {args.compare}")
            return
        episode_metrics_b = [compute_episode_metrics(ep) for ep in episodes_b]
        agg_b = aggregate_metrics(episode_metrics_b)
        print_metrics(str(args.compare), agg_b)
        print_comparison(str(args.state_logs_dir), agg, str(args.compare), agg_b)

    # Optionally save
    if args.output:
        output = {"primary": agg, "per_episode": episode_metrics}
        if args.compare:
            output["compare"] = agg_b
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
