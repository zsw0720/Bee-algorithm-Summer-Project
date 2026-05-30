# main.py

import sys
import os
import config

# Reconfigure encoding for Windows console to handle UTF-8 inputs correctly
if sys.platform.startswith("win"):
    try:
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older python versions
        import io
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from vlm_parser import parse_image
from coords_transformer import CoordinateTransformer
from bees_algorithm import NDTPathPlannerBA
from simulation import visualize_ndt_plan

def print_banner():
    print("=" * 75)
    print("      VLM-Guided Bees Algorithm Automated NDT Path Planning System")
    print("=" * 75)
    print("This system uses:")
    print("1. A Vision-Language Model (VLM) to analyze NDT photographs and locate welds.")
    print("2. A Camera Coordinate Transformer to map pixels to physical dimensions.")
    print("3. The Bees Algorithm to optimize a collision-free robot scan path.")
    print("4. Matplotlib to overlay the animated scanning path directly on the image.")
    print("=" * 75)
    print(f"Current Settings:")
    print(f"- VLM Mode: {config.LLM_MODE.upper()}")
    if config.LLM_MODE == "gemini":
        print(f"  - Model: {config.GEMINI_MODEL}")
        print(f"  - API Key: {'Configured' if config.GEMINI_API_KEY else 'Missing (using local fallback)'}")
    print(f"- Default Inspection Image: {config.IMAGE_PATH} ({config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT} px)")
    print(f"- Scout Bees (n): {config.BA_N}")
    print(f"- Waypoints per segment (M): {config.BA_WAYPOINTS_COUNT}")
    print("=" * 75)

def main():
    print_banner()

    # Create Coordinate Transformer helper
    transformer = CoordinateTransformer(
        img_width=config.IMAGE_WIDTH,
        img_height=config.IMAGE_HEIGHT,
        map_width=config.MAP_WIDTH,
        map_height=config.MAP_HEIGHT
    )

    # Interactive Command Loop
    while True:
        print("\n[Input Options]")
        print("1. Run VLM Visual Detection on default image (metal_plate.png)")
        print("2. Enter a custom image file path for inspection")
        print("3. Toggle VLM Mode (Local Mock <-> Gemini VLM API)")
        print("4. Start Interactive Web Dashboard Server")
        print("5. Exit")
        choice = input("Select an option (1-5): ").strip()

        if choice == "5":
            print("Exiting NDT System. Goodbye!")
            sys.exit(0)

        elif choice == "4":
            print("\nStarting Interactive Web Dashboard Server...")
            print("Access the dashboard at: http://127.0.0.1:5000")
            print("Press Ctrl+C to stop the server.")
            import subprocess
            try:
                subprocess.run([sys.executable, "server.py"])
            except KeyboardInterrupt:
                print("\nServer stopped.")
            continue

        elif choice == "3":
            if config.LLM_MODE == "local":
                config.LLM_MODE = "gemini"
                # If key is missing, prompt user
                if not config.GEMINI_API_KEY:
                    key = input("Enter your Gemini API Key (or press enter to skip): ").strip()
                    if key:
                        config.GEMINI_API_KEY = key
            else:
                config.LLM_MODE = "local"
            print(f"LLM Mode toggled to: {config.LLM_MODE.upper()}")
            continue

        elif choice == "1":
            image_path = config.IMAGE_PATH
            print(f"\nAnalyzing default NDT image asset: '{image_path}'")
        
        elif choice == "2":
            image_path = input("\nEnter the path to your custom image:\n> ").strip()
            if not image_path:
                print("Image path cannot be empty!")
                continue
            if not os.path.exists(image_path):
                print(f"Error: File '{image_path}' does not exist!")
                continue
        else:
            print("Invalid option selected!")
            continue

        # Check if target image asset is present
        if not os.path.exists(image_path):
            print(f"\n[Error] Inspection image '{image_path}' not found in the workspace!")
            print("Please make sure the image asset is placed in the project directory.")
            continue

        # Step 1: Parse the NDT image using VLM
        parsed_data = parse_image(image_path)
        pixel_targets = parsed_data.get("targets", [])
        target_names = parsed_data.get("target_names", [])
        weld_boxes = parsed_data.get("weld_boxes", [])
        pixel_obstacles = parsed_data.get("obstacles", [])

        # Check if Ground Truth mask exists for VLM detection accuracy evaluation
        base_path, ext = os.path.splitext(image_path)
        gt_mask_path = base_path + "_gt" + ext
        # Fallback to .png extension for ground truth if the image has a different extension
        if not os.path.exists(gt_mask_path):
            gt_mask_path = base_path + "_gt.png"
            
        if os.path.exists(gt_mask_path) and weld_boxes:
            print("\n[Evaluation] Ground Truth mask detected. Triggering accuracy evaluation...")
            from evaluation import evaluate_vlm_accuracy
            evaluate_vlm_accuracy(gt_mask_path, weld_boxes, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

        # Step 2: Convert pixel coordinates to physical robot coordinates
        physics_targets = [transformer.pixel_to_physics(u, v) for u, v in pixel_targets]
        physics_obstacles = [transformer.pixel_box_to_physics_rect(box[0], box[1], box[2], box[3]) for box in pixel_obstacles]

        print("\n--- Visual Recognition Results (Pixel Space) ---")
        print(f"Detected {len(pixel_targets)} weld waypoints:")
        for idx, (name, pt) in enumerate(zip(target_names, pixel_targets)):
            print(f"  {idx+1}. {name}: Pixel {pt}")
        if weld_boxes:
            print(f"Detected {len(weld_boxes)} weld bounding boxes:")
            for idx, box in enumerate(weld_boxes):
                print(f"  Box {idx+1}: Bounding Box {box}")
        print(f"Detected {len(pixel_obstacles)} obstacles:")
        for idx, box in enumerate(pixel_obstacles):
            print(f"  Obstacle {idx+1}: Bounding Box {box}")
        print("-------------------------------------------------")

        print("\n--- Camera Calibration & Coordinate Mapping ---")
        print(f"Mapped coordinate transformation matrix:")
        print(f"  Pixel Width  {config.IMAGE_WIDTH} -> Physical Width  {config.MAP_WIDTH}mm")
        print(f"  Pixel Height {config.IMAGE_HEIGHT} -> Physical Height {config.MAP_HEIGHT}mm")
        print(f"Mapped targets in physical scanning space:")
        for idx, (name, coord) in enumerate(zip(target_names, physics_targets)):
            print(f"  {idx+1}. {name}: Physical Coord ({coord[0]:.2f}, {coord[1]:.2f}) mm")
        print("-------------------------------------------------")

        # Step 3: Establish the path segments sequence
        # Sequence: Start -> Target 1 -> Target 2 -> ... -> Target N -> End
        sequence = [config.START_POS] + physics_targets + [config.END_POS]
        
        path_segments = []
        convergence_histories = []
        total_optimized_length = 0.0
        success = True

        print("\nOptimizing path segments with Bees Algorithm...")
        # Optimize each segment sequentially
        for i in range(len(sequence) - 1):
            seg_start = sequence[i]
            seg_end = sequence[i+1]
            print(f"  Segment {i+1}: ({seg_start[0]:.2f}, {seg_start[1]:.2f}) -> ({seg_end[0]:.2f}, {seg_end[1]:.2f}) mm")
            
            planner = NDTPathPlannerBA(seg_start, seg_end, physics_obstacles)
            result = planner.optimize()
            
            path_segments.append(result["path"])
            convergence_histories.append(result["fitness_history"])
            total_optimized_length += result["length"]
            
            if not result["is_valid"]:
                print(f"  [Warning] Segment {i+1} failed to find a completely collision-free path.")
                success = False
            else:
                print(f"    Segment {i+1} optimized successfully. Length: {result['length']:.2f} mm")

        print("\n--- Trajectory Planning Summary ---")
        print(f"Total optimized trajectory length: {total_optimized_length:.2f} mm")
        if success:
            print("Status: SUCCESS (All segments collision-free)")
        else:
            print("Status: WARNING (Collision detected, adjust BA parameters)")
        print("-----------------------------------")

        # Step 4: Run simulation and overlay path on top of the NDT image
        print("\nLaunching VLM NDT path animation window...")
        visualize_ndt_plan(physics_targets, target_names, physics_obstacles, path_segments, convergence_histories, image_path=image_path)

if __name__ == "__main__":
    main()
