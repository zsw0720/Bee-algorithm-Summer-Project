# config.py

import os

# Load environment variables from .env file if it exists
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# ==========================================
# 1. Physical Environment & NDT Settings
# ==========================================
# Map size (2D Plate bounds in mm)
MAP_WIDTH = 100.0
MAP_HEIGHT = 100.0

# VLM Image Assets configurations
IMAGE_PATH = "metal_plate.png"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# Start position for NDT crawler robot
START_POS = (5.0, 5.0)
END_POS = (95.0, 5.0)

# Predefined Target Regions (Welds)
TARGET_REGIONS = {
    "left_weld": {
        "x_range": (21.5, 25.5),
        "y_range": (20.0, 80.0),
        "color": "#99ccff"
    },
    "right_weld": {
        "x_range": (76.8, 80.8),
        "y_range": (20.0, 80.0),
        "color": "#99ccff"
    }
}

# Static Obstacles (Safety Valves, Holes, etc.)
# Represented as [x_min, y_min, x_max, y_max]
OBSTACLES = [
    [34.0, 34.0, 67.0, 67.0],  # Center safety valve with safety margin
]

# ==========================================
# 2. Bees Algorithm (BA) Hyperparameters
# ==========================================
BA_N = 40          # Number of scout bees (initial population)
BA_M = 15          # Number of selected patches/sites (out of n)
BA_E = 5           # Number of elite patches/sites (out of m)
BA_NEP = 25        # Number of recruited bees for elite patches
BA_NSP = 12        # Number of recruited bees for non-elite selected patches
BA_NGH = 8.0       # Initial neighborhood/patch size (radius of search)
BA_NGH_SHRINK = 0.98  # Shrink factor for neighborhood size when no improvement
BA_MAX_IT = 80     # Maximum number of iterations
BA_WAYPOINTS_COUNT = 3  # Number of intermediate waypoints to optimize for each path segment

# ==========================================
# 3. LLM Configuration
# ==========================================
# "local" for offline mock parsing, "gemini" for online Google Gemini API
LLM_MODE = "gemini" 

# Gemini API settings
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
