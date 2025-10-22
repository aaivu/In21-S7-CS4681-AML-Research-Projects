import numpy as np

CLASS_IOU_THRESH = {
    'Car': 0.7,
    'Pedestrian': 0.5,
    'Cyclist': 0.5
}

DIFFICULTY = ['Easy', 'Moderate', 'Hard']

def iou_3d(box_a, box_b):
    """Axis-aligned 3D IoU for boxes [x, y, z, dx, dy, dz, heading]"""
    max_a = box_a[:3] + box_a[3:6]/2
    min_a = box_a[:3] - box_a[3:6]/2
    max_b = box_b[:3] + box_b[3:6]/2
    min_b = box_b[:3] - box_b[3:6]/2

    overlap_min = np.maximum(min_a, min_b)
    overlap_max = np.minimum(max_a, max_b)
    overlap = np.maximum(overlap_max - overlap_min, 0)
    inter_vol = np.prod(overlap)
    vol_a = np.prod(box_a[3:6])
    vol_b = np.prod(box_b[3:6])
    return inter_vol / (vol_a + vol_b - inter_vol + 1e-6)

def iou_bev(box_a, box_b):
    """Bird's-eye view IoU for boxes [x, y, z, dx, dy, dz, heading]"""
    max_a = box_a[:2] + box_a[3:5]/2
    min_a = box_a[:2] - box_a[3:5]/2
    max_b = box_b[:2] + box_b[3:5]/2
    min_b = box_b[:2] - box_b[3:5]/2

    overlap_min = np.maximum(min_a, min_b)
    overlap_max = np.minimum(max_a, max_b)
    overlap = np.maximum(overlap_max - overlap_min, 0)
    inter_area = np.prod(overlap)
    area_a = box_a[3] * box_a[4]
    area_b = box_b[3] * box_b[4]
    return inter_area / (area_a + area_b - inter_area + 1e-6)

def compute_ap(recall, precision):
    """KITTI 40-point AP interpolation"""
    recall_points = np.linspace(0, 1, 40)
    ap = 0.0
    for r in recall_points:
        p = precision[recall >= r].max() if np.any(recall >= r) else 0
        ap += p / 40
    return ap

def evaluate_predictions(pred_boxes, pred_labels, gt_boxes, gt_labels):
    """
    Compute AP3D and APBEV per class.
    pred_boxes: [N,7], pred_labels: [N]
    gt_boxes: [M,7], gt_labels: [M]
    """
    results = {}
    for cls, iou_thresh in CLASS_IOU_THRESH.items():
        pred_mask = pred_labels == cls
        gt_mask = gt_labels == cls

        pred_cls_boxes = pred_boxes[pred_mask]
        gt_cls_boxes = gt_boxes[gt_mask]

        if pred_cls_boxes.shape[0] == 0 or gt_cls_boxes.shape[0] == 0:
            results[cls] = {'3d': [0]*3, 'bev':[0]*3}
            continue

        tp_3d = []
        tp_bev = []

        # Dummy scores; you can extend to actual confidence scores
        for pb in pred_cls_boxes:
            ious3d = np.array([iou_3d(pb, gb) for gb in gt_cls_boxes])
            iousbev = np.array([iou_bev(pb, gb) for gb in gt_cls_boxes])
            tp_3d.append(1 if ious3d.max() >= iou_thresh else 0)
            tp_bev.append(1 if iousbev.max() >= iou_thresh else 0)

        tp_3d = np.array(tp_3d)
        tp_bev = np.array(tp_bev)
        fp_3d = 1 - tp_3d
        fp_bev = 1 - tp_bev

        # Precision-Recall
        cum_tp_3d = np.cumsum(tp_3d)
        cum_fp_3d = np.cumsum(fp_3d)
        recall_3d = cum_tp_3d / (len(gt_cls_boxes)+1e-6)
        precision_3d = cum_tp_3d / (cum_tp_3d + cum_fp_3d + 1e-6)

        cum_tp_bev = np.cumsum(tp_bev)
        cum_fp_bev = np.cumsum(fp_bev)
        recall_bev = cum_tp_bev / (len(gt_cls_boxes)+1e-6)
        precision_bev = cum_tp_bev / (cum_tp_bev + cum_fp_bev + 1e-6)

        results[cls] = {
            '3d': [compute_ap(recall_3d, precision_3d)]*3,
            'bev':[compute_ap(recall_bev, precision_bev)]*3
        }

    return results
