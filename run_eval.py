"""
Example script for running 10 rollouts of a DROID policy on the example environment.

Usage:

First, make sure you download the simulation assets and unpack them into the root directory of this package.

Then, in a separate terminal, launch the policy server on localhost:8000 
-- make sure to set XLA_PYTHON_CLIENT_MEM_FRACTION to avoid JAX hogging all the GPU memory.

For example, to launch a pi0-FAST-DROID policy (with joint position control), 
run the command below in a separate terminal from the openpi "karl/droid_policies" branch:

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos

Finally, run the evaluation script:

python run_eval.py --episodes 10 --headless
"""

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from sim_evals.inference.droid_jointpos import Client as DroidJointPosClient


def check_task_success(env, scene: int) -> bool:
    """
    Check if the task was successfully completed based on object positions.

    Scene 1: Cube in bowl - check if cube is inside bowl bounds
    Scene 2: Can in mug - check if can is inside mug bounds
    Scene 3: Banana in bin - check if banana is inside bin bounds
    """
    try:
        # Define object pairs for each scene (target_object, container_object)
        scene_objects = {
            1: ("Cube", "Bowl"),
            2: ("Can", "Mug"),
            3: ("Banana", "Bin"),
        }

        if scene not in scene_objects:
            return False

        target_name, container_name = scene_objects[scene]

        # Get object positions from environment scene
        # Objects are registered as rigid bodies in the scene
        if hasattr(env.env.scene, target_name) and hasattr(env.env.scene, container_name):
            target = env.env.scene[target_name]
            container = env.env.scene[container_name]

            # Get positions (center of mass)
            target_pos = target.data.root_pos_w[0].cpu().numpy()  # [x, y, z]
            container_pos = container.data.root_pos_w[0].cpu().numpy()  # [x, y, z]

            # Check if target is inside container (simple distance + height check)
            # Object is "in" container if:
            # 1. Horizontal distance is small (within container radius)
            # 2. Height is close to or below container height

            horizontal_dist = ((target_pos[0] - container_pos[0])**2 +
                             (target_pos[1] - container_pos[1])**2)**0.5
            height_diff = target_pos[2] - container_pos[2]

            # Success criteria (adjust these thresholds as needed)
            HORIZONTAL_THRESHOLD = 0.15  # 15cm horizontal tolerance
            HEIGHT_THRESHOLD_MIN = -0.05  # Can be slightly below container
            HEIGHT_THRESHOLD_MAX = 0.20   # Can be up to 20cm above container (just dropped in)

            is_inside = (horizontal_dist < HORIZONTAL_THRESHOLD and
                        HEIGHT_THRESHOLD_MIN < height_diff < HEIGHT_THRESHOLD_MAX)

            if is_inside:
                print(f"  ✓ Success detected: {target_name} is in {container_name}")
                print(f"    Horizontal distance: {horizontal_dist:.3f}m, Height diff: {height_diff:.3f}m")

            return is_inside
        else:
            # Objects not found in scene, fall back to termination flag
            print(f"  Warning: Could not find {target_name} or {container_name} in scene")
            return False

    except Exception as e:
        print(f"  Warning: Error checking task success: {e}")
        return False


def main(
        episodes:int = 10,
        headless: bool = True,
        scene: int = 1,
        ):
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg


    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = None
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "put the can in the mug"
        case 3:
            instruction = "put banana in the bin"
        case _:
            raise ValueError(f"Scene {scene} not supported")
        
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials
    client = DroidJointPosClient()


    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)

    # Create state_logs directory for robot sensor data
    state_logs_dir = video_dir / "state_logs"
    state_logs_dir.mkdir(parents=True, exist_ok=True)

    video = []
    ep = 0
    max_steps = env.env.max_episode_length
    success_count = 0  # Track successful episodes

    with torch.no_grad():
        for ep in range(episodes):
            # Initialize episode state log
            episode_states = []
            step_count = 0
            terminated = False
            truncated = False

            for step_idx in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]

                # Log robot state (before taking action)
                robot_state = obs["policy"]
                state_entry = {
                    "step": step_idx,
                    "arm_joint_pos": robot_state["arm_joint_pos"][0].cpu().numpy().tolist(),
                    "gripper_pos": robot_state["gripper_pos"][0].cpu().numpy().tolist(),
                    "action": ret["action"].tolist(),
                }
                episode_states.append(state_entry)

                obs, _, term, trunc, _ = env.step(action)
                step_count += 1
                terminated = bool(term)
                truncated = bool(trunc)

                if term or trunc:
                    break

            # Check actual task success based on object positions
            task_success = check_task_success(env, scene)

            # Save episode state log
            state_log_file = state_logs_dir / f"episode_{ep}_state.json"
            state_log_data = {
                "episode": ep,
                "instruction": instruction,
                "scene": scene,
                "num_steps": step_count,
                "terminated": terminated,
                "truncated": truncated,
                "success": task_success,  # TRUE success based on object positions
                "states": episode_states,
            }

            with open(state_log_file, 'w') as f:
                json.dump(state_log_data, f, indent=2)

            # Update success counter
            if task_success:
                success_count += 1

            success_str = "✓ SUCCESS" if task_success else "✗ FAILED"
            print(f"  Episode {ep}: {success_str} ({step_count} steps)")
            print(f"  Saved state log: {state_log_file.name}")

            client.reset()
            mediapy.write_video(
                video_dir / f"episode_{ep}.mp4",
                video,
                fps=15,
            )
            video = []

    # Print final summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Scene: {scene} - \"{instruction}\"")
    print(f"Episodes: {episodes}")
    print(f"Success rate: {success_count}/{episodes} ({100*success_count/episodes:.1f}%)")
    print(f"Results saved to: {video_dir}")
    print("=" * 70)

    # Save summary file
    summary_file = video_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "scene": scene,
            "instruction": instruction,
            "total_episodes": episodes,
            "successful_episodes": success_count,
            "success_rate": success_count / episodes if episodes > 0 else 0,
        }, f, indent=2)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
