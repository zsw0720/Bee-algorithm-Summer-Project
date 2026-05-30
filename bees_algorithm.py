# bees_algorithm.py

import os
import importlib.util
import numpy as np
import config

# Dynamically import the official BeesAlgorithm class to avoid name collisions with this file
official_module_path = os.path.join(
    os.path.dirname(__file__), 
    "bees_algorithm_official", 
    "bees_algorithm", 
    "bees_algorithm.py"
)

spec = importlib.util.spec_from_file_location("official_bees_algorithm", official_module_path)
official_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official_module)

# Expose the BeesAlgorithm class from the official repository
BeesAlgorithm = official_module.BeesAlgorithm

# ==========================================
# 1. Collision Detection Helpers
# ==========================================

def ccw(A, B, C):
    """
    Checks if points A, B, C are in counter-clockwise order.
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def line_segments_intersect(A, B, C, D):
    """
    Checks if line segment AB intersects line segment CD.
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def point_inside_rectangle(pt, rect):
    """
    Checks if point pt=(x, y) is inside the rectangle rect=[x_min, y_min, x_max, y_max].
    """
    x, y = pt
    x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max

def segment_intersects_rectangle(p1, p2, rect):
    """
    Checks if line segment from p1 to p2 intersects the rectangular obstacle.
    """
    if point_inside_rectangle(p1, rect) or point_inside_rectangle(p2, rect):
        return True

    x_min, y_min, x_max, y_max = rect
    bottom_left  = (x_min, y_min)
    bottom_right = (x_max, y_min)
    top_left     = (x_min, y_max)
    top_right    = (x_max, y_max)

    if line_segments_intersect(p1, p2, bottom_left, bottom_right): # Bottom edge
        return True
    if line_segments_intersect(p1, p2, top_left, top_right):       # Top edge
        return True
    if line_segments_intersect(p1, p2, bottom_left, top_left):     # Left edge
        return True
    if line_segments_intersect(p1, p2, bottom_right, top_right):   # Right edge
        return True

    return False


# ==========================================
# 2. Path Optimization Class (Wrapping Official BA)
# ==========================================

class NDTPathPlannerBA:
    def __init__(self, start_pt, end_pt, obstacles):
        """
        Initializes the planner for a single path segment from start_pt to end_pt.
        """
        self.start = start_pt
        self.end = end_pt
        self.obstacles = obstacles
        
        self.num_waypoints = config.BA_WAYPOINTS_COUNT
        self.dim = self.num_waypoints * 2  # [x1, y1, x2, y2, ..., xM, yM]
        
        # Plate bounds
        self.x_min, self.x_max = 0.0, config.MAP_WIDTH
        self.y_min, self.y_max = 0.0, config.MAP_HEIGHT

    def reconstruct_path(self, solution):
        """
        Converts a flat solution vector into a list of coordinate pairs:
        [Start, Waypoint1, Waypoint2, ..., WaypointM, End]
        """
        path = [self.start]
        for i in range(self.num_waypoints):
            x = solution[2 * i]
            y = solution[2 * i + 1]
            path.append((x, y))
        path.append(self.end)
        return path

    def calculate_fitness(self, solution):
        """
        Evaluates a solution. 
        Fitness is 1.0 / (Total Length + Collision Penalty).
        """
        path = self.reconstruct_path(solution)
        
        # 1. Calculate path length
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.sqrt(dx**2 + dy**2)
            
        # 2. Check for collisions
        collisions = 0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            for obs in self.obstacles:
                if segment_intersects_rectangle(p1, p2, obs):
                    collisions += 1

        # Penalize collisions heavily
        penalty = collisions * 5000.0
        total_cost = length + penalty
        return 1.0 / (total_cost + 1e-6)

    def optimize(self):
        """
        Runs Professor Pham's official Bees Algorithm to optimize the waypoints.
        """
        # Define ranges for all variables
        range_min = []
        range_max = []
        for _ in range(self.num_waypoints):
            range_min.extend([self.x_min, self.y_min])
            range_max.extend([self.x_max, self.y_max])

        # Convert absolute config.BA_NGH to relative neighborhood search scale for each dimension
        # In official code, local search generates new values by adding np.random.uniform(-v, v) * ngh[i]
        # where v = (range_max[i] - range_min[i]) * 0.5 (which is 50.0 for 100 range).
        # We want the max perturbation to be config.BA_NGH.
        # So: v * initial_ngh[i] = config.BA_NGH => 50.0 * initial_ngh[i] = BA_NGH => initial_ngh[i] = BA_NGH / 50.0
        rel_ngh = [ (2.0 * config.BA_NGH) / (range_max[i] - range_min[i]) for i in range(self.dim) ]

        # Instantiate official BeesAlgorithm
        # We map our configs to official parameters:
        # ns -> config.BA_N (scouts)
        # nb -> config.BA_M (selected sites)
        # ne -> config.BA_E (elite sites)
        # nrb -> config.BA_NSP (selected foragers)
        # nre -> config.BA_NEP (elite foragers)
        # stlim -> 10 (stagnation limit for site abandonment)
        # shrink_factor -> calculated from config.BA_NGH_SHRINK
        alg = BeesAlgorithm(
            score_function=self.calculate_fitness,
            range_min=range_min,
            range_max=range_max,
            ns=config.BA_N,
            nb=config.BA_M,
            ne=config.BA_E,
            nrb=config.BA_NSP,
            nre=config.BA_NEP,
            stlim=10,
            initial_ngh=rel_ngh,
            shrink_factor=1.0 - config.BA_NGH_SHRINK
        )

        fitness_history = []
        
        # Run optimization step-by-step to record fitness history
        for _ in range(config.BA_MAX_IT):
            best_score = alg.performSingleStep()
            fitness_history.append(best_score)

        # Retrieve optimized solution
        best_solution = alg.best_solution.values
        optimized_waypoints = self.reconstruct_path(best_solution)

        # Calculate final length
        final_length = 0.0
        for i in range(len(optimized_waypoints) - 1):
            dx = optimized_waypoints[i+1][0] - optimized_waypoints[i][0]
            dy = optimized_waypoints[i+1][1] - optimized_waypoints[i][1]
            final_length += np.sqrt(dx**2 + dy**2)

        # Check for collisions in the final path
        collisions = 0
        for i in range(len(optimized_waypoints) - 1):
            for obs in self.obstacles:
                if segment_intersects_rectangle(optimized_waypoints[i], optimized_waypoints[i+1], obs):
                    collisions += 1

        is_valid = (collisions == 0)

        return {
            "path": optimized_waypoints,
            "length": final_length,
            "is_valid": is_valid,
            "fitness_history": fitness_history
        }
