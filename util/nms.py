import torch
from torchvision.ops import nms

def perform_nms(pred_boxes, scores, img_container):
    # Distinguish between rings and segments based on vertical coverage
    img_height = img_container.boxes.shape[1] # assuming boxes are in (N, 4) format with (x1, y1, x2, y2)
    y_extent = pred_boxes[:, 3] - pred_boxes[:, 1]
    is_ring = y_extent >= img_height * 0.35 # heuristic: rings cover at least 35% of vertical axis, segments less
    
    # Separate boxes into rings and segments
    ring_mask = is_ring
    segment_mask = ~is_ring
    
    ring_boxes = pred_boxes[ring_mask]
    ring_scores = scores[ring_mask]
    segment_boxes = pred_boxes[segment_mask]
    segment_scores = scores[segment_mask]
    
    # Apply NMS to rings with lenient threshold
    ring_idx_keep = []
    if len(ring_boxes) > 0:
        ring_idx_keep = nms(ring_boxes, ring_scores, iou_threshold=0.1)
    
    # Apply NMS to segments with strict threshold
    segment_idx_keep = []
    if len(segment_boxes) > 0:
        segment_idx_keep = nms(segment_boxes, segment_scores, iou_threshold=0.4)
    
    # Reconstruct boxes and scores
    filtered_boxes = []
    filtered_scores = []
    
    if len(ring_idx_keep) > 0:
        filtered_boxes.append(ring_boxes[ring_idx_keep])
        filtered_scores.append(ring_scores[ring_idx_keep])

    if len(segment_idx_keep) > 0:
        filtered_boxes.append(segment_boxes[segment_idx_keep])
        filtered_scores.append(segment_scores[segment_idx_keep])
    
    if filtered_boxes:
        return torch.cat(filtered_boxes), torch.cat(filtered_scores)
    else:
        return torch.empty((0, 4), device=pred_boxes.device), torch.empty((0,), device=scores.device)