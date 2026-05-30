# simulation.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import config

def visualize_ndt_plan(targets, target_names, obstacles, path_segments, convergence_histories, image_path=None):
    """
    Visualizes the planned path, animates the scanner, and plots the Bees Algorithm convergence.
    """
    # 1. Prepare data
    # Flatten the path segments into a single list of waypoints for plotting the full path
    full_path = []
    for idx, seg in enumerate(path_segments):
        if idx == 0:
            full_path.extend(seg)
        else:
            full_path.extend(seg[1:])  # Avoid repeating connection nodes (e.g. Target 1)

    # Calculate smooth intermediate points for the animation
    anim_points = []
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i+1]
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        # Generate points proportional to the distance for constant speed
        num_steps = max(3, int(dist * 1.5))
        for t in np.linspace(0, 1, num_steps, endpoint=False):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            anim_points.append((x, y))
    anim_points.append(full_path[-1])

    # 2. Set up the figure with 2 subplots (left: map & path, right: convergence)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    plt.suptitle("LLM-Bees Algorithm NDT Path Planning System", fontsize=16, fontweight='bold')

    # ==========================================
    # SUBPLOT 1: Map and Path Plotting
    # ==========================================
    ax1.set_xlim(0, config.MAP_WIDTH)
    ax1.set_ylim(0, config.MAP_HEIGHT)
    ax1.set_xlabel("X Coordinate (mm)", fontweight='bold')
    ax1.set_ylabel("Y Coordinate (mm)", fontweight='bold')
    ax1.set_title("VLM-Guided NDT Scanning Simulation", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 1. Load and Draw Background Image (if exists)
    import os
    from PIL import Image
    bg_img = None
    target_img_path = image_path if image_path is not None else config.IMAGE_PATH
    if target_img_path and os.path.exists(target_img_path):
        try:
            bg_img = Image.open(target_img_path)
            ax1.imshow(bg_img, extent=[0, config.MAP_WIDTH, 0, config.MAP_HEIGHT], alpha=0.85, zorder=1)
            print(f"[Simulation] Loaded background image '{target_img_path}' successfully using Pillow.")
        except Exception as e:
            print(f"[Warning] Failed to load background image: {e}")

    # 2. Draw Welds (Target Regions) - only if no background image is present
    if bg_img is None:
        for name, region in config.TARGET_REGIONS.items():
            xr = region["x_range"]
            yr = region["y_range"]
            rect = plt.Rectangle((xr[0], yr[0]), xr[1] - xr[0], yr[1] - yr[0], 
                                 fc=region["color"], ec='blue', lw=1.5, alpha=0.4, 
                                 label="Weld Region" if name == "left_weld" else "")
            ax1.add_patch(rect)
            ax1.text((xr[0]+xr[1])/2, (yr[0]+yr[1])/2, name.replace("_", " ").title(), 
                     color='blue', weight='bold', rotation=90, fontsize=9, ha='center', va='center')

    # 3. Draw Obstacles (No-Go Zones)
    all_obstacles = obstacles if obstacles is not None else config.OBSTACLES
    for idx, obs in enumerate(all_obstacles):
        alpha_val = 0.3 if bg_img is not None else 0.7
        rect = plt.Rectangle((obs[0], obs[1]), obs[2] - obs[0], obs[3] - obs[1], 
                             fc='#ff9999', ec='red', lw=2, alpha=alpha_val, 
                             label="Obstacle (No-Go Zone)" if idx == 0 else "", zorder=2)
        ax1.add_patch(rect)
        if bg_img is None:
            ax1.text((obs[0]+obs[2])/2, (obs[1]+obs[3])/2, "Obstacle", 
                     color='red', weight='bold', fontsize=9, ha='center', va='center')

    # Draw Target waypoints extracted by LLM
    tx = [pt[0] for pt in targets]
    ty = [pt[1] for pt in targets]
    ax1.scatter(tx, ty, color='darkblue', edgecolor='black', s=80, zorder=5, label='Target Waypoints (LLM)')
    for i, (name, pt) in enumerate(zip(target_names, targets)):
        ax1.text(pt[0] + 2, pt[1] + 1, f"T{i+1}: {name}", fontsize=8, color='darkblue', weight='bold')

    # Draw Start and End positions
    ax1.scatter(config.START_POS[0], config.START_POS[1], color='gold', edgecolor='black', marker='*', s=150, zorder=5, label='Start')
    ax1.scatter(config.END_POS[0], config.END_POS[1], color='orange', edgecolor='black', marker='X', s=100, zorder=5, label='End')

    # Draw Path
    px = [pt[0] for pt in full_path]
    py = [pt[1] for pt in full_path]
    ax1.plot(px, py, color='#2ca02c', linestyle='-', linewidth=2.5, alpha=0.8, label='Optimized Path (BA)', zorder=4)

    # Animate Probe scanning along path
    probe, = ax1.plot([], [], color='black', marker='o', markersize=8, markeredgecolor='white', zorder=6, label='NDT Probe')
    probe_trail, = ax1.plot([], [], color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.6, zorder=3)

    ax1.legend(loc='upper right', framealpha=0.9, facecolor='white', fontsize='small')

    # ==========================================
    # SUBPLOT 2: Bees Algorithm Convergence Curve
    # ==========================================
    ax2.set_xlabel("Iteration", fontweight='bold')
    ax2.set_ylabel("Best Fitness (1/Cost)", fontweight='bold')
    ax2.set_title("Bees Algorithm Optimization Convergence", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)

    for idx, history in enumerate(convergence_histories):
        # We plot the cost instead of raw fitness for easier interpretation (Cost = 1/Fitness)
        costs = [1.0 / fit for fit in history]
        ax2.plot(costs, label=f"Segment {idx+1}")
        
    ax2.set_yscale('log') # Log scale is better to see convergence of large costs
    ax2.legend(loc='upper right', framealpha=0.9, fontsize='small')

    # ==========================================
    # 3. Animation Logic
    # ==========================================
    def init():
        probe.set_data([], [])
        probe_trail.set_data([], [])
        return probe, probe_trail

    def update(frame):
        x, y = anim_points[frame]
        probe.set_data([x], [y])
        
        # Draw trail
        tx = [pt[0] for pt in anim_points[:frame+1]]
        ty = [pt[1] for pt in anim_points[:frame+1]]
        probe_trail.set_data(tx, ty)
        return probe, probe_trail

    # Create the animation object
    ani = animation.FuncAnimation(
        fig, update, frames=len(anim_points), init_func=init,
        interval=40, blit=True, repeat=True
    )

    plt.tight_layout()
    # Save a static copy of the finalized plot
    plt.savefig('d:/Summer Project/ndt_latest_run_result.png', dpi=300)
    print("Static summary image saved to 'd:/Summer Project/ndt_latest_run_result.png'")

    # Show the plot window interactively
    plt.show()
