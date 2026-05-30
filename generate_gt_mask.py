# generate_gt_mask.py
import numpy as np
from PIL import Image

# Create a binary image of 1024x1024 pixels (L mode for grayscale/binary)
width, height = 1024, 1024
mask = np.zeros((height, width), dtype=np.uint8)

# 1. Define True Left Weld Region (Ground Truth)
# Calibrated X coordinate center at u = 235 px -> range [210, 260]
# Y coordinate maps to v in [200, 800] px
mask[200:800, 210:260] = 255

# 2. Define True Right Weld Region (Ground Truth)
# Calibrated X coordinate center at u = 788 px -> range [763, 813]
# Y coordinate maps to v in [200, 800] px
mask[200:800, 763:813] = 255

# Save mask as PNG
img = Image.fromarray(mask)
img.save("metal_plate_gt.png")
print("Calibrated Ground Truth mask successfully saved as 'metal_plate_gt.png'")
