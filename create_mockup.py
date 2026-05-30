import matplotlib.pyplot as plt
import numpy as np

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Set grid and labels
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel('X Coordinate (mm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y Coordinate (mm)', fontsize=12, fontweight='bold')
ax.set_title('Simulation Environment Mockup:\nLLM-Guided NDT Path Planning with Bees Algorithm', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, linestyle='--', alpha=0.5)

# 1. Plot Obstacle (Central Safety Valve)
obstacle = plt.Rectangle((40, 40), 20, 20, fc='#ff9999', ec='red', lw=2, alpha=0.8, label='Obstacle (Safety Valve)')
ax.add_patch(obstacle)
ax.text(50, 50, 'Obstacle\n(No-Go Zone)', color='red', weight='bold', fontsize=10, ha='center', va='center')

# 2. Plot NDT Inspection Targets (Welding Seams)
# Left Weld
left_weld = plt.Rectangle((10, 20), 10, 60, fc='#99ccff', ec='blue', lw=1.5, alpha=0.6, label='NDT Target (Welding Seams)')
ax.add_patch(left_weld)
ax.text(15, 50, 'Left Weld', color='blue', weight='bold', rotation=90, fontsize=10, ha='center', va='center')

# Right Weld
right_weld = plt.Rectangle((80, 20), 10, 60, fc='#99ccff', ec='blue', lw=1.5, alpha=0.6)
ax.add_patch(right_weld)
ax.text(85, 50, 'Right Weld', color='blue', weight='bold', rotation=90, fontsize=10, ha='center', va='center')

# 3. Plot specific inspection target points extracted by LLM
targets_x = [15, 15, 85]
targets_y = [50, 75, 25]
ax.scatter(targets_x, targets_y, color='darkblue', edgecolor='black', s=100, zorder=5, label='Target Waypoints (extracted by LLM)')
for i, (tx, ty) in enumerate(zip(targets_x, targets_y)):
    ax.text(tx + 2, ty + 1, f'Target {i+1}', fontsize=9, color='darkblue', weight='bold')

# 4. Plot start position
start_x, start_y = 5, 5
ax.scatter(start_x, start_y, color='gold', edgecolor='black', marker='*', s=200, zorder=5, label='Robot Start Position')
ax.text(start_x + 2, start_y + 1, 'Start', fontsize=10, color='darkgoldenrod', weight='bold')

# 5. Plot Path planned by Bees Algorithm (avoiding obstacle)
# Waypoints of the path: Start -> Target 1 -> Target 2 -> (bypass obstacle by going above and right) -> Target 3 -> End
path_x = [5, 15, 15, 35, 65, 75, 85, 95]
path_y = [5, 50, 75, 70, 70, 45, 25, 5]

ax.plot(path_x, path_y, color='#2ca02c', linestyle='-', linewidth=3, alpha=0.9, label='Bees Algorithm Optimized Path', zorder=4)
# Add arrows to indicate direction
for i in range(len(path_x) - 1):
    dx = (path_x[i+1] - path_x[i]) * 0.5
    dy = (path_y[i+1] - path_y[i]) * 0.5
    ax.annotate('', xy=(path_x[i] + dx, path_y[i] + dy), xytext=(path_x[i], path_y[i]),
                arrowprops=dict(arrowstyle="->", color='#1e7b1e', lw=2), size=10)

# Add Legend
ax.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='gray')

# Add semantic input text box
instruction_text = (
    "Human Language Input (LLM):\n"
    "\"First scan the middle and top of the left weld,\n"
    "then scan the lower part of the right weld.\n"
    "Avoid the safety valve in the center.\""
)
props = dict(boxstyle='round', facecolor='#e6f2ff', alpha=0.9, edgecolor='#b3d9ff', lw=1.5)
ax.text(3, 97, instruction_text, transform=ax.transData, fontsize=10,
        verticalalignment='top', bbox=props, family='sans-serif')

plt.tight_layout()
plt.savefig('d:/Summer Project/ndt_simulation_mockup.png', dpi=300)
print("Updated mockup image successfully saved to d:/Summer Project/ndt_simulation_mockup.png")
