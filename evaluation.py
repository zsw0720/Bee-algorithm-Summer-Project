# evaluation.py

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def evaluate_vlm_accuracy(gt_mask_path, predicted_boxes, image_shape=(600, 800)):
    """
    Computes Precision, Recall, IoU, and F1-Score by comparing VLM predicted 
    bounding boxes against the Ground Truth binary mask.
    Generates a 3-panel comparison plot.
    """
    print(f"\n--- Running VLM Detection Accuracy Evaluation ---")
    print(f"Loading Ground Truth mask from: '{gt_mask_path}'...")
    
    if not os.path.exists(gt_mask_path):
        print(f"[Error] Ground Truth mask file '{gt_mask_path}' not found!")
        return None

    # 1. Load Ground Truth binary mask
    try:
        gt_img = Image.open(gt_mask_path).convert("L")
        gt_mask = np.array(gt_img) > 127  # Threshold to binary boolean array
    except Exception as e:
        print(f"[Error] Failed to load GT mask: {e}")
        return None

    # 2. Construct Predicted Mask from Bounding Boxes
    pred_mask = np.zeros(image_shape, dtype=bool)
    height, width = image_shape
    
    for idx, box in enumerate(predicted_boxes):
        u_min, v_min, u_max, v_max = box
        # Clip coordinates to image boundary
        u_min_idx = max(0, int(round(u_min)))
        u_max_idx = min(width, int(round(u_max)))
        v_min_idx = max(0, int(round(v_min)))
        v_max_idx = min(height, int(round(v_max)))
        
        pred_mask[v_min_idx:v_max_idx, u_min_idx:u_max_idx] = True
        print(f"  Mapping predicted Weld Box {idx+1}: Pixel u[{u_min_idx}:{u_max_idx}], v[{v_min_idx}:{v_max_idx}] to prediction mask.")

    # 3. Calculate Metrics (Pixel-level)
    TP_mask = pred_mask & gt_mask   # True Positive (Correctly detected defect pixels)
    FP_mask = pred_mask & ~gt_mask  # False Positive (False alarm pixels)
    FN_mask = ~pred_mask & gt_mask  # False Negative (Missed defect pixels)

    TP = float(np.sum(TP_mask))
    FP = float(np.sum(FP_mask))
    FN = float(np.sum(FN_mask))

    intersection = TP
    union = TP + FP + FN

    # Calculations
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    iou = intersection / union if union > 0 else 1.0
    f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 3b. Calculate Box-level IoU (Standard Object Detection metric)
    box_iou = 0.0
    y_indices, x_indices = np.where(gt_mask)
    if len(x_indices) > 0 and len(y_indices) > 0 and predicted_boxes:
        gt_box = [float(np.min(x_indices)), float(np.min(y_indices)), float(np.max(x_indices)), float(np.max(y_indices))]
        
        # Merge all predicted boxes into one bounding box for union assessment
        pred_x_min = min([box[0] for box in predicted_boxes])
        pred_y_min = min([box[1] for box in predicted_boxes])
        pred_x_max = max([box[2] for box in predicted_boxes])
        pred_y_max = max([box[3] for box in predicted_boxes])
        pred_box = [pred_x_min, pred_y_min, pred_x_max, pred_y_max]
        
        # Compute Box Intersection
        ix_min = max(gt_box[0], pred_box[0])
        iy_min = max(gt_box[1], pred_box[1])
        ix_max = min(gt_box[2], pred_box[2])
        iy_max = min(gt_box[3], pred_box[3])
        
        iw = max(0.0, ix_max - ix_min)
        ih = max(0.0, iy_max - iy_min)
        box_intersection = iw * ih
        
        # Compute Box Areas and Union
        area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        box_union = area_gt + area_pred - box_intersection
        
        box_iou = box_intersection / box_union if box_union > 0 else 0.0
        print(f"Box-Level Overlap: GT Box={gt_box}, Pred Box={pred_box}, Box IoU={box_iou*100:.2f}%")
    else:
        print("Could not compute Box-level IoU (either no GT pixels or no predicted boxes).")

    print("\n--- Accuracy Evaluation Report ---")
    print(f"Pixel Counts:")
    print(f"  - True Positives (Correct): {int(TP)} px")
    print(f"  - False Positives (False Alarm): {int(FP)} px")
    print(f"  - False Negatives (Missed): {int(FN)} px")
    print(f"Evaluation Metrics:")
    print(f"  - Precision (Precision Rate): {precision:.4f} ({precision*100:.2f}%)")
    print(f"  - Recall (Detection Rate):    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  - Pixel-level IoU:            {iou:.4f} ({iou*100:.2f}%)")
    print(f"  - Box-level IoU:              {box_iou:.4f} ({box_iou*100:.2f}%)")
    print(f"  - F1-Score (Dice Coefficient): {f1_score:.4f} ({f1_score*100:.2f}%)")
    print("-----------------------------------")

    # 4. Generate Visual Overlay Plot
    viz_img = np.zeros((height, width, 3), dtype=np.uint8)
    viz_img[TP_mask] = [0, 220, 0]    # Green
    viz_img[FP_mask] = [220, 0, 0]    # Red
    viz_img[FN_mask] = [0, 0, 220]    # Blue

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    plt.suptitle("VLM Defect Detection Accuracy Evaluation", fontsize=14, fontweight='bold')

    # Plot 1: Ground Truth
    ax1.imshow(gt_mask, cmap='gray')
    ax1.set_title("Ground Truth Mask\n(True Defects)", fontsize=10, fontweight='bold')
    ax1.axis('off')

    # Plot 2: VLM Predicted Mask
    ax2.imshow(pred_mask, cmap='gray')
    ax2.set_title("VLM Predicted Mask\n(from Bounding Boxes)", fontsize=10, fontweight='bold')
    ax2.axis('off')

    # Plot 3: Error Visual Overlay
    ax3.imshow(viz_img)
    ax3.set_title("Overlay Comparison\n(Green: TP, Red: FP, Blue: FN)", fontsize=10, fontweight='bold')
    ax3.axis('off')

    # Add text box with metrics
    metric_text = (
        f"Pixel IoU: {iou*100:.1f}%\n"
        f"Box IoU: {box_iou*100:.1f}%\n"
        f"F1-Score: {f1_score*100:.1f}%\n"
        f"Precision: {precision*100:.1f}%\n"
        f"Recall: {recall*100:.1f}%"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax3.text(10, height - 10, metric_text, fontsize=9,
             verticalalignment='bottom', horizontalalignment='left', bbox=props)

    plt.tight_layout()
    plot_path = "d:/Summer Project/ndt_evaluation_result.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Visual evaluation comparison saved to '{plot_path}'")
    plt.close('all') # Make sure we close plot to avoid thread warning/leaks

    return {
        "iou": iou,
        "box_iou": box_iou,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall
    }
