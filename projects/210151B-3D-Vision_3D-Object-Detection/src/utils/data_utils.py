import numpy as np

def load_kitti_labels(label_file):
    """
    Load KITTI 3D labels, handle incomplete lines
    """
    boxes = []
    labels = []

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:  # skip incomplete lines
                continue
            class_name = parts[0]
            labels.append({'Car':0, 'Pedestrian':1, 'Cyclist':2}.get(class_name, 0))
            # bbox: x, y, z, h, w, l, r
            try:
                x, y, z, h, w, l, r = map(float, parts[11:18])
            except:
                x = y = z = h = w = l = r = 0.0
            boxes.append([x, y, z, h, w, l, r])

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
