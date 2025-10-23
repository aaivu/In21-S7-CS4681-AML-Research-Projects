import torch
import numpy as np
import os
from src.models.pv_rcnn_pp import PVRCNNPP_Simple
from src.train import points_to_voxel_tensor

def load_pointcloud(pc_file):
    """Load a KITTI .bin point cloud"""
    points = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
    voxels = points_to_voxel_tensor(points)
    return voxels.squeeze(0)

def main():
    # Paths
    base_dir = r"C:\Users\Prasanna De  Silva\Desktop\SEM 7\Adv. ML\PV_RCNN\data\kitti"
    velodyne_dir = os.path.join(base_dir, "train", "velodyne")  # or test folder

    # Initialize model
    model = PVRCNNPP_Simple()
    checkpoint = r"experiments/logs/pvrcnnpp_simple.pt"
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()

    # Loop through all .bin files
    for f in sorted(os.listdir(velodyne_dir)):
        if f.endswith(".bin"):
            pc_file = os.path.join(velodyne_dir, f)
            voxels = load_pointcloud(pc_file)
            voxels = voxels.unsqueeze(0)  # add batch dimension

            with torch.no_grad():
                cls_logits, box_reg = model(voxels, voxels)

            # Convert logits to predicted class
            pred_labels = torch.argmax(cls_logits, dim=1)

            print(f"File: {f}")
            print(f"Predicted labels shape: {pred_labels.shape}")
            print(f"Predicted boxes shape: {box_reg.shape}")
            print("-" * 50)

if __name__ == "__main__":
    main()
