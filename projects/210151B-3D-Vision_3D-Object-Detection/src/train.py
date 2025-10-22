import torch
import numpy as np

def points_to_voxel_tensor(points, grid_size=(32, 32, 8)):
    """
    Convert point cloud N x 4 -> voxel tensor [C=4, D, H, W]
    Channels: mean x, mean y, mean z, point density
    """
    if points.shape[1] == 3:
        # if reflectance missing, add dummy intensity channel
        reflectance = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack((points, reflectance))

    x, y, z, r = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    W, H, D = grid_size
    x_idx = ((x - x.min()) / (x.max() - x.min() + 1e-6) * (W - 1)).astype(int)
    y_idx = ((y - y.min()) / (y.max() - y.min() + 1e-6) * (H - 1)).astype(int)
    z_idx = ((z - z.min()) / (z.max() - z.min() + 1e-6) * (D - 1)).astype(int)

    # initialize [C=4, D, H, W]
    voxel = np.zeros((4, D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    for i in range(points.shape[0]):
        dx, dy, dz = z_idx[i], y_idx[i], x_idx[i]
        voxel[0, dx, dy, dz] += x[i]
        voxel[1, dx, dy, dz] += y[i]
        voxel[2, dx, dy, dz] += z[i]
        voxel[3, dx, dy, dz] += r[i]
        count[dx, dy, dz] += 1

    # avoid divide-by-zero
    nonzero = count > 0
    voxel[:, nonzero] /= count[nonzero]

    # normalize density to [0,1]
    voxel[3] = voxel[3] / (voxel[3].max() + 1e-6)
    voxel_tensor = torch.from_numpy(voxel)  # [4, D, H, W]
    # voxel_tensor = torch.from_numpy(voxel).unsqueeze(0)  # [1, 4, D, H, W]
    return voxel_tensor
