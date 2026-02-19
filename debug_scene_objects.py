"""
Debug script to inspect what objects are actually available in the sim-eval environment.

This will help us figure out the correct way to access object positions.

Usage:
    cd ../sim-evals
    uv run python debug_scene_objects.py --scene 1
"""

import tyro
import argparse
import gymnasium as gym
import torch
from pathlib import Path


def main(scene: int = 1):
    # Launch omniverse app
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import after app launch
    import sim_evals.environments
    from isaaclab_tasks.utils import parse_env_cfg

    # Initialize env
    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()

    print("\n" + "=" * 70)
    print(f"DEBUGGING SCENE {scene}")
    print("=" * 70)

    # 1. Check what's in env.env.scene
    print("\n[1] Checking env.env.scene attributes:")
    print(f"    Type: {type(env.env.scene)}")
    print(f"    Dir: {[x for x in dir(env.env.scene) if not x.startswith('_')][:20]}")

    # 2. Try to list all scene entities
    print("\n[2] Trying to list scene entities:")
    if hasattr(env.env.scene, '__dict__'):
        scene_attrs = [k for k in env.env.scene.__dict__.keys() if not k.startswith('_')]
        print(f"    Scene attributes: {scene_attrs}")

    # 3. Check if we can iterate through scene
    print("\n[3] Trying to access scene entities:")
    try:
        # Try different access patterns
        if hasattr(env.env.scene, 'rigid_objects'):
            print("    ✓ env.env.scene.rigid_objects exists")
            print(f"      Type: {type(env.env.scene.rigid_objects)}")

        if hasattr(env.env.scene, 'articulations'):
            print("    ✓ env.env.scene.articulations exists")

        # Try to access known objects by guessing names
        potential_names = [
            "Cube", "cube", "CUBE", "Cube_01",
            "Bowl", "bowl", "BOWL", "Bowl_01",
            "Can", "can", "CAN",
            "Mug", "mug", "MUG",
            "Banana", "banana", "BANANA",
            "Bin", "bin", "BIN",
            "scene/Cube", "scene/Bowl",
        ]

        print("\n[4] Trying to access objects by name:")
        for name in potential_names:
            try:
                obj = env.env.scene[name]
                print(f"    ✓ Found: '{name}'")
                print(f"      Type: {type(obj)}")
                if hasattr(obj, 'data'):
                    print(f"      Has 'data' attribute")
                    if hasattr(obj.data, 'root_pos_w'):
                        pos = obj.data.root_pos_w[0].cpu().numpy()
                        print(f"      Position: {pos}")
            except (KeyError, AttributeError, IndexError) as e:
                pass  # Object not found or wrong attribute

    except Exception as e:
        print(f"    ✗ Error: {e}")

    # 4. Check observation structure
    print("\n[5] Observation structure:")
    print(f"    obs keys: {obs.keys()}")
    if "policy" in obs:
        print(f"    obs['policy'] keys: {obs['policy'].keys()}")

    # 5. Try to access env info
    print("\n[6] Environment info:")
    if hasattr(env.env, 'scene'):
        if hasattr(env.env.scene, 'cfg'):
            print(f"    Scene config: {env.env.scene.cfg}")

    # 6. Check if there's a way to get all rigid bodies
    print("\n[7] Trying to find all rigid bodies:")
    try:
        # IsaacLab typically has a scene manager
        scene = env.env.scene

        # Try to iterate through all registered entities
        if hasattr(scene, '_data'):
            print(f"    scene._data type: {type(scene._data)}")

        # Check class attributes
        for attr_name in dir(scene):
            if not attr_name.startswith('_'):
                attr = getattr(scene, attr_name)
                if hasattr(attr, 'data') and hasattr(attr.data, 'root_pos_w'):
                    try:
                        pos = attr.data.root_pos_w[0].cpu().numpy()
                        print(f"    ✓ '{attr_name}' has position: {pos}")
                    except:
                        pass

    except Exception as e:
        print(f"    ✗ Error: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Based on the output above, you can determine:
1. How to access scene objects (env.env.scene[name] or other?)
2. The correct object names (Cube vs cube vs CUBE?)
3. How to get positions (data.root_pos_w or something else?)

Use this information to fix the check_task_success() function in run_eval.py
    """)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
