import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.models.pv_rcnn_pp import PVRCNNPP
from src.train import points_to_voxel_tensor
from src.utils.data_utils import load_kitti_labels
import numpy as np


class KittiDataset(Dataset):
    """
    Custom Dataset for KITTI 3D Object Detection
    """
    def __init__(self, lidar_dir, label_dir, max_samples=None):
        self.lidar_dir = os.path.abspath(lidar_dir)
        self.label_dir = os.path.abspath(label_dir)

        if not os.path.exists(self.lidar_dir):
            raise FileNotFoundError(f"LIDAR directory not found: {self.lidar_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.files = []
        for f in sorted(os.listdir(self.lidar_dir)):
            if f.endswith('.bin'):
                label_file = os.path.join(self.label_dir, f.replace('.bin', '.txt'))
                if os.path.exists(label_file):
                    self.files.append(f)
                else:
                    print(f"⚠️ Warning: Missing label file for {f}, skipping.")

        if len(self.files) == 0:
            raise RuntimeError(f"No matching KITTI files found.")

        if max_samples is not None:
            self.files = self.files[:max_samples]
            print(f"📦 Using only {len(self.files)} samples for this run.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pc_file = os.path.join(self.lidar_dir, self.files[idx])
        label_file = os.path.join(self.label_dir, self.files[idx].replace('.bin', '.txt'))

        points = np.fromfile(pc_file, dtype=np.float32)
        if points.size % 4 == 0:
            points = points.reshape(-1, 4)
        elif points.size % 3 == 0:
            # add dummy reflectance column
            points = points.reshape(-1, 3)
            reflectance = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.hstack((points, reflectance))
        else:
            raise ValueError(f"Unexpected point cloud shape in {pc_file}")

        
        voxels = points_to_voxel_tensor(points)  # should return [C, D, H, W]


        try:
            boxes, labels = load_kitti_labels(label_file)
        except ValueError:
            # Skip bad label files
            boxes = np.zeros((1, 7), dtype=np.float32)
            labels = np.zeros((1,), dtype=np.int64)

        return voxels.float(), torch.from_numpy(points).float(), torch.from_numpy(labels).long(), torch.from_numpy(boxes).float()


def main():
    # Project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(__file__))

    # KITTI dataset paths relative to project root
    base_dir = os.path.join(project_root, "data", "kitti")

    dataset = KittiDataset(
        lidar_dir=os.path.join(base_dir, "train", "velodyne"),
        label_dir=os.path.join(base_dir, "train", "label_2"),
        max_samples=7480
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PVRCNNPP(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_box = torch.nn.MSELoss()
      
    for epoch in range(10):
        for voxels, points, labels, boxes in loader:
            voxels = voxels.to(device)
            points = points.to(device)
            labels = labels.to(device)
            boxes = boxes.to(device)
            
            optimizer.zero_grad()
            cls_logits, box_reg = model(voxels, points)  # fixed forward

            labels_int = labels.view(-1)
            boxes_first = boxes[:, 0, :] if boxes.shape[1] > 0 else boxes

            min_len = min(cls_logits.shape[0], labels_int.shape[0])
            loss_cls = criterion_cls(cls_logits[:min_len], labels_int[:min_len])
            loss_box = criterion_box(box_reg[:min_len], boxes_first[:min_len])

            loss = loss_cls + loss_box
            loss.backward()
            optimizer.step()

        print(f"✅ Epoch {epoch} completed")

    # Save model relative to project root
    log_dir = os.path.join(project_root, 'experiments', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(log_dir, 'pvrcnnpp_fixed.pt'))
    print("🎉 Training completed and model saved.")


if __name__ == "__main__":
    main()