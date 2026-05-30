# server.py
import os
import sys

# Crucial: Set matplotlib backend to non-interactive Agg before importing other modules
# This prevents crashes when running inside Flask's background threads.
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import config
from vlm_parser import parse_image
from coords_transformer import CoordinateTransformer
from bees_algorithm import NDTPathPlannerBA
from evaluation import evaluate_vlm_accuracy

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Keep track of the currently loaded image path and its dimensions
current_state = {
    "image_path": config.IMAGE_PATH,
    "width": config.IMAGE_WIDTH,
    "height": config.IMAGE_HEIGHT
}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Serve uploaded images and results
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/inspect', methods=['POST'])
def inspect_image():
    if 'image' not in request.files:
        # If no image uploaded, check if we want to run default image
        use_default = request.form.get('use_default', 'false') == 'true'
        if use_default:
            image_path = config.IMAGE_PATH
            filename = os.path.basename(image_path)
            # Copy to upload directory so frontend can access it via /uploads/
            import shutil
            dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(dest) and os.path.exists(image_path):
                shutil.copy(image_path, dest)
            filepath = image_path
        else:
            return jsonify({"error": "No image file provided"}), 400
    else:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if user also uploaded a Ground Truth mask
        gt_file = request.files.get('gt_mask')
        if gt_file:
            gt_filename = secure_filename(gt_file.filename)
            # Ensure it aligns with filepath naming convention (base_gt.png)
            base_name, _ = os.path.splitext(filename)
            gt_filename = f"{base_name}_gt.png"
            gt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gt_filename)
            gt_file.save(gt_filepath)

    # Inspect image size using PIL
    from PIL import Image
    try:
        with Image.open(filepath) as img:
            img_w, img_h = img.size
    except Exception as e:
        return jsonify({"error": f"Failed to open image file: {str(e)}"}), 400

    # Save to state
    current_state["image_path"] = filepath
    current_state["width"] = img_w
    current_state["height"] = img_h

    # Toggle VLM mode in config dynamically
    vlm_mode = request.form.get('vlm_mode', 'local').lower()
    config.LLM_MODE = 'gemini' if vlm_mode == 'gemini' else 'local'

    # Step 1: Parse features
    user_prompt = request.form.get('user_prompt', '')
    parsed_data = parse_image(filepath, user_prompt=user_prompt)
    pixel_targets = parsed_data.get("targets", [])
    target_names = parsed_data.get("target_names", [])
    weld_boxes = parsed_data.get("weld_boxes", [])
    pixel_obstacles = parsed_data.get("obstacles", [])

    # Step 2: VLM Evaluation (if Ground Truth exists)
    metrics = None
    
    # Check for Ground Truth mask file in uploads or same directory
    base_path, ext = os.path.splitext(filepath)
    gt_mask_path = base_path + "_gt" + ext
    if not os.path.exists(gt_mask_path):
        gt_mask_path = base_path + "_gt.png"
    # Fallback to local workspace if uploaded inside project directory
    if not os.path.exists(gt_mask_path):
        local_base = os.path.splitext(os.path.basename(filepath))[0]
        gt_mask_path = local_base + "_gt.png"

    if os.path.exists(gt_mask_path) and weld_boxes:
        try:
            print(f"[API] Evaluating defect detection accuracy against GT mask '{gt_mask_path}'...")
            metrics = evaluate_vlm_accuracy(gt_mask_path, weld_boxes, (img_h, img_w))
        except Exception as e:
            print(f"[Warning] Accuracy evaluation failed: {e}")

    # Step 3: Convert pixel coordinates to physical robot coordinates
    transformer = CoordinateTransformer(
        img_width=img_w,
        img_height=img_h,
        map_width=config.MAP_WIDTH,
        map_height=config.MAP_HEIGHT
    )
    physics_targets = [transformer.pixel_to_physics(u, v) for u, v in pixel_targets]
    physics_obstacles = [transformer.pixel_box_to_physics_rect(box[0], box[1], box[2], box[3]) for box in pixel_obstacles]

    # Serve the uploaded file name
    web_image_url = f"/uploads/{os.path.basename(filepath)}"
    web_gt_url = f"/uploads/{os.path.basename(gt_mask_path)}" if os.path.exists(gt_mask_path) else None
    
    # If the image was loaded from default workspace path, serve from uploads
    if filepath == config.IMAGE_PATH:
        web_image_url = f"/uploads/{os.path.basename(config.IMAGE_PATH)}"

    return jsonify({
        "success": True,
        "image_url": web_image_url,
        "gt_url": web_gt_url,
        "width": img_w,
        "height": img_h,
        "vlm_mode": config.LLM_MODE.upper(),
        "pixel_targets": pixel_targets,
        "target_names": target_names,
        "weld_boxes": weld_boxes,
        "pixel_obstacles": pixel_obstacles,
        "physics_targets": physics_targets,
        "physics_obstacles": physics_obstacles,
        "metrics": metrics
    })

@app.route('/api/plan', methods=['POST'])
def plan_path():
    data = request.json or {}
    
    # 1. Retrieve & Override Bees Algorithm Parameters
    config.BA_N = int(data.get('n', config.BA_N))
    config.BA_M = int(data.get('m', config.BA_M))
    config.BA_E = int(data.get('e', config.BA_E))
    config.BA_NEP = int(data.get('nep', config.BA_NEP))
    config.BA_NSP = int(data.get('nsp', config.BA_NSP))
    config.BA_MAX_IT = int(data.get('max_it', config.BA_MAX_IT))
    config.BA_WAYPOINTS_COUNT = int(data.get('waypoints_count', config.BA_WAYPOINTS_COUNT))
    config.BA_NGH = float(data.get('ngh', config.BA_NGH))
    
    start_pos = tuple(data.get('start_pos', config.START_POS))
    end_pos = tuple(data.get('end_pos', config.END_POS))
    
    physics_targets = data.get('physics_targets', [])
    physics_obstacles = data.get('physics_obstacles', [])
    
    # Sequence of segments: Start -> Target 1 -> ... -> Target N -> End
    sequence = [start_pos] + [tuple(pt) for pt in physics_targets] + [end_pos]
    
    path_segments = []
    convergence_histories = []
    total_length = 0.0
    overall_success = True
    
    print(f"[API] Optimizing trajectory path using Bees Algorithm (n={config.BA_N}, iter={config.BA_MAX_IT})...")
    
    for i in range(len(sequence) - 1):
        seg_start = sequence[i]
        seg_end = sequence[i+1]
        
        planner = NDTPathPlannerBA(seg_start, seg_end, physics_obstacles)
        result = planner.optimize()
        
        path_segments.append(result["path"])
        convergence_histories.append(result["fitness_history"])
        total_length += result["length"]
        
        if not result["is_valid"]:
            overall_success = False

    # Also map physical paths back to pixel coordinates for the frontend canvas to draw
    transformer = CoordinateTransformer(
        img_width=current_state["width"],
        img_height=current_state["height"],
        map_width=config.MAP_WIDTH,
        map_height=config.MAP_HEIGHT
    )
    
    pixel_path_segments = []
    for seg in path_segments:
        pixel_seg = [transformer.physics_to_pixel(pt[0], pt[1]) for pt in seg]
        pixel_path_segments.append(pixel_seg)

    return jsonify({
        "success": True,
        "total_length": total_length,
        "is_valid": overall_success,
        "physics_path_segments": path_segments,
        "pixel_path_segments": pixel_path_segments,
        "convergence_histories": convergence_histories
    })

if __name__ == '__main__':
    # Clean up matplotlib figures to save memory
    import matplotlib.pyplot as plt
    plt.close('all')
    app.run(host='127.0.0.1', port=5000, debug=True)
