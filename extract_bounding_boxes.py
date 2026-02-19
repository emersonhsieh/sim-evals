"""
Script to extract accurate bounding box dimensions from Isaac Sim scenes.

This will query the actual rigid body extents from the simulation at runtime,
giving us the real container dimensions for success detection.

Usage:
    cd ../sim-evals
    uv run python extract_bounding_boxes.py
"""

import tyro
import argparse
import gymnasium as gym
import json
from pathlib import Path


def get_bounding_box(rigid_body):
    """
    Extract bounding box dimensions from a rigid body.

    Returns:
        dict: Contains 'center', 'extents', 'min', 'max' in world coordinates
    """
    try:
        print(f"    [get_bounding_box] Getting position...")
        # Get position (center of mass)
        pos = rigid_body.data.root_pos_w[0].cpu().numpy()
        print(f"    [get_bounding_box] Position: {pos}")

        # Try to get bounding box extents
        # In Isaac Sim, this might be in different attributes
        bbox_info = {
            "center": pos.tolist(),
        }

        # Check various possible attributes for bounding box
        print(f"    [get_bounding_box] Checking for body_aabb_w...")
        if hasattr(rigid_body.data, 'body_aabb_w'):
            print(f"    [get_bounding_box] Found body_aabb_w, extracting...")
            # AABB (Axis-Aligned Bounding Box) in world frame
            # Shape: (num_envs, num_bodies, 6) where 6 = [min_x, min_y, min_z, max_x, max_y, max_z]
            aabb_all = rigid_body.data.body_aabb_w[0].cpu().numpy()  # (num_bodies, 6)
            print(f"    [get_bounding_box] body_aabb_w shape (after env index): {aabb_all.shape}")

            # Compute union AABB across all bodies
            aabb_min = aabb_all[:, :3].min(axis=0)  # min of all body mins
            aabb_max = aabb_all[:, 3:].max(axis=0)  # max of all body maxes

            bbox_info["aabb_min"] = aabb_min.tolist()
            bbox_info["aabb_max"] = aabb_max.tolist()

            # Calculate extents (half-widths)
            extents = (aabb_max - aabb_min) / 2
            bbox_info["extents"] = extents.tolist()

            # Calculate dimensions
            dimensions = aabb_max - aabb_min
            bbox_info["dimensions"] = dimensions.tolist()
            print(f"    [get_bounding_box] Extracted dimensions: {dimensions}")

        else:
            print(f"    [get_bounding_box] No body_aabb_w, checking root_physx_view...")
            if hasattr(rigid_body, 'root_physx_view'):
                # Try to get from PhysX view
                view = rigid_body.root_physx_view
                print(f"    [get_bounding_box] Found root_physx_view, checking get_aabbs...")
                if hasattr(view, 'get_aabbs'):
                    print(f"    [get_bounding_box] Calling get_aabbs()...")
                    aabb = view.get_aabbs()[0]
                    bbox_info["aabb_min"] = aabb[:3].tolist()
                    bbox_info["aabb_max"] = aabb[3:].tolist()

                    extents = (aabb[3:] - aabb[:3]) / 2
                    bbox_info["extents"] = extents.tolist()
                    dimensions = aabb[3:] - aabb[:3]
                    bbox_info["dimensions"] = dimensions.tolist()
                    print(f"    [get_bounding_box] Extracted dimensions: {dimensions}")
                else:
                    print(f"    [get_bounding_box] No get_aabbs method")
            else:
                print(f"    [get_bounding_box] No root_physx_view")

        # Fallback: compute bounding box from USD stage geometry
        if 'dimensions' not in bbox_info:
            print(f"    [get_bounding_box] No AABB from physics, trying USD BBoxCache...")
            try:
                import omni.usd
                from pxr import UsdGeom, Usd

                stage = omni.usd.get_context().get_stage()
                # Resolve prim path: replace {ENV_REGEX_NS} with actual env path
                prim_path = rigid_body.cfg.prim_path.replace(
                    "{ENV_REGEX_NS}", "/World/envs/env_0"
                )
                print(f"    [get_bounding_box] Looking up USD prim: {prim_path}")
                prim = stage.GetPrimAtPath(prim_path)

                if prim.IsValid():
                    bbox_cache = UsdGeom.BBoxCache(
                        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
                    )
                    bbox = bbox_cache.ComputeWorldBound(prim)
                    bbox_range = bbox.ComputeAlignedRange()
                    bbox_min = bbox_range.GetMin()
                    bbox_max = bbox_range.GetMax()

                    aabb_min = [bbox_min[0], bbox_min[1], bbox_min[2]]
                    aabb_max = [bbox_max[0], bbox_max[1], bbox_max[2]]
                    dimensions = [aabb_max[i] - aabb_min[i] for i in range(3)]
                    extents = [d / 2 for d in dimensions]

                    bbox_info["aabb_min"] = aabb_min
                    bbox_info["aabb_max"] = aabb_max
                    bbox_info["dimensions"] = dimensions
                    bbox_info["extents"] = extents
                    print(f"    [get_bounding_box] USD dimensions: {dimensions}")
                else:
                    print(f"    [get_bounding_box] Prim not valid at {prim_path}")
            except Exception as usd_err:
                print(f"    [get_bounding_box] USD fallback failed: {usd_err}")

        # If we have geometry data
        print(f"    [get_bounding_box] Checking for mass...")
        if hasattr(rigid_body, 'data') and hasattr(rigid_body.data, 'default_mass'):
            bbox_info["mass"] = float(rigid_body.data.default_mass[0])
            print(f"    [get_bounding_box] Mass: {bbox_info['mass']}")

        print(f"    [get_bounding_box] Done, returning bbox_info")
        return bbox_info

    except Exception as e:
        print(f"    [get_bounding_box] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


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

    # Container objects for each scene
    scene_containers = {
        1: ("rubiks_cube", "_24_bowl"),
        2: ("_10_potted_meat_can", "_25_mug"),
        3: ("_11_banana", "small_KLT_visual_collision"),
    }

    all_bbox_data = {}

    print(f"\n{'='*70}")
    print(f"EXTRACTING BOUNDING BOXES FOR SCENE {scene}")
    print('='*70)

    # Initialize env for this scene
    print(f"[DEBUG] Step 1: Parsing env config...")
    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    print(f"[DEBUG] Step 2: Setting scene to {scene}...")
    env_cfg.set_scene(scene)
    print(f"[DEBUG] Step 3: Creating gym environment...")
    env = gym.make("DROID", cfg=env_cfg)
    print(f"[DEBUG] Step 4: Environment created successfully")

    print(f"[DEBUG] Step 5: First reset...")
    obs, _ = env.reset()
    print(f"[DEBUG] Step 6: First reset completed")

    print(f"[DEBUG] Step 7: Second reset (for materials)...")
    obs, _ = env.reset()
    print(f"[DEBUG] Step 8: Second reset completed")

    target_name, container_name = scene_containers[scene]
    print(f"[DEBUG] Step 9: Target={target_name}, Container={container_name}")

    if True:  # Keep indentation for compatibility
        scene_num = scene

        print(f"\nScene {scene_num} objects:")
        print(f"  Target: {target_name}")
        print(f"  Container: {container_name}")

        scene_data = {
            "scene": scene_num,
            "target_name": target_name,
            "container_name": container_name,
        }

        # Get rigid objects
        print(f"[DEBUG] Step 10: Checking for rigid_objects...")
        if hasattr(env.env.scene, 'rigid_objects'):
            print(f"[DEBUG] Step 11: Found rigid_objects, accessing...")
            rigid_objects = env.env.scene.rigid_objects
            print(f"[DEBUG] Step 12: Available objects: {list(rigid_objects.keys())}")

            # Extract target bounding box
            print(f"[DEBUG] Step 13: Extracting target ({target_name}) bounding box...")
            if target_name in rigid_objects:
                target = rigid_objects[target_name]
                print(f"[DEBUG] Step 14: Got target object, calling get_bounding_box...")
                target_bbox = get_bounding_box(target)
                print(f"[DEBUG] Step 15: Target bounding box extracted")
                scene_data["target_bbox"] = target_bbox

                print(f"\n{target_name}:")
                print(f"  Center: {target_bbox.get('center')}")
                if 'dimensions' in target_bbox:
                    dims = target_bbox['dimensions']
                    print(f"  Dimensions (L×W×H): {dims[0]:.3f}m × {dims[1]:.3f}m × {dims[2]:.3f}m")
                if 'extents' in target_bbox:
                    ext = target_bbox['extents']
                    print(f"  Half-extents: {ext[0]:.3f}m × {ext[1]:.3f}m × {ext[2]:.3f}m")
            else:
                print(f"  ✗ Could not find {target_name}")

            # Extract container bounding box
            print(f"[DEBUG] Step 16: Extracting container ({container_name}) bounding box...")
            if container_name in rigid_objects:
                container = rigid_objects[container_name]
                print(f"[DEBUG] Step 17: Got container object, calling get_bounding_box...")
                container_bbox = get_bounding_box(container)
                print(f"[DEBUG] Step 18: Container bounding box extracted")
                scene_data["container_bbox"] = container_bbox

                print(f"\n{container_name}:")
                print(f"  Center: {container_bbox.get('center')}")
                if 'dimensions' in container_bbox:
                    dims = container_bbox['dimensions']
                    print(f"  Dimensions (L×W×H): {dims[0]:.3f}m × {dims[1]:.3f}m × {dims[2]:.3f}m")

                    # Calculate suggested thresholds
                    # Horizontal threshold should be ~radius of container opening
                    horizontal_extent = max(dims[0], dims[1]) / 2
                    print(f"  Suggested horizontal threshold: {horizontal_extent:.3f}m")

                    # Height threshold based on container height
                    height = dims[2]
                    print(f"  Suggested height range: -{height/2:.3f}m to +{height/2:.3f}m")

                if 'extents' in container_bbox:
                    ext = container_bbox['extents']
                    print(f"  Half-extents: {ext[0]:.3f}m × {ext[1]:.3f}m × {ext[2]:.3f}m")
            else:
                print(f"  ✗ Could not find {container_name}")

    all_bbox_data[f"scene_{scene_num}"] = scene_data

    print(f"[DEBUG] Step 19: Closing environment...")
    env.close()
    print(f"[DEBUG] Step 20: Environment closed successfully")

    # Save bounding box data
    output_file = Path(f"bounding_boxes_scene{scene}.json")
    with open(output_file, 'w') as f:
        json.dump(all_bbox_data, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Bounding box data saved to: {output_file}")
    print("\nRecommended updates for check_task_success():")

    scene_key = f"scene_{scene_num}"
    if scene_key in all_bbox_data:
        scene_data = all_bbox_data[scene_key]
        container_bbox = scene_data.get("container_bbox", {})

        if 'dimensions' in container_bbox:
            dims = container_bbox['dimensions']
            horizontal_threshold = max(dims[0], dims[1]) / 2
            height_threshold = dims[2] / 2

            print(f"\nScene {scene_num} ({scene_data['container_name']}):")
            print(f"  HORIZONTAL_THRESHOLD = {horizontal_threshold:.3f}  # Container radius")
            print(f"  HEIGHT_THRESHOLD_MIN = -{height_threshold:.3f}  # Bottom of container")
            print(f"  HEIGHT_THRESHOLD_MAX = +{height_threshold:.3f}  # Top of container")
        else:
            print(f"\nNote: Could not extract dimensions for scene {scene}")
            print("Center positions shown above can still help validate thresholds")

    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
