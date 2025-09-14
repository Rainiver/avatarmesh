import numpy as np
from scipy.spatial.transform import Rotation as R
import uuid
import os

# Define your 21 c2w (camera-to-world) matrices
c2w_matrices = np.array([
    [[-0.2948, -0.1659, 0.9411, 3.7642],
     [0.9556, -0.0512, 0.2903, 1.1611],
     [0.0000, 0.9848, 0.1736, 0.6946],
     [0.0000, 0.0000, 0.0000, 1.0000]],
    [[-0.5633, -0.1435, 0.8137, 3.2547],
     [0.8262, -0.0978, 0.5548, 2.2190],
     [0.0000, 0.9848, 0.1736, 0.6946],
     [0.0000, 0.0000, 0.0000, 1.0000]],
    ...
    [[0.0000, -0.1736, 0.9848, 3.9392],
     [1.0000, 0.0000, 0.0000, 0.0000],
     [0.0000, 0.9848, 0.1736, 0.6946],
     [0.0000, 0.0000, 0.0000, 1.0000]]
])

# Alternative transformation option (commented out)
# c2w_tmp = c2w_matrices.copy()
# c2w_matrices[:, :3, 1] = c2w_tmp[:, :3, 2]
# c2w_matrices[:, :3, 2] = c2w_tmp[:, :3, 1]
# c2w_matrices[:, 1, 3] = c2w_tmp[:, 2, 3]
# c2w_matrices[:, 2, 3] = c2w_tmp[:, 1, 3]
# cam_poses = c2w_matrices
# poses = np.array(cam_poses)
# x_axis = poses[:, :3, 0]
# y_axis = poses[:, :3, 1]
# z_axis = poses[:, :3, 2]
# center = poses[:, :3, 3]
# SAVE_ROOT = '.'
# np.savetxt(os.path.join(SAVE_ROOT, f"cam_z.txt"),
# np.concatenate([center, z_axis], axis=-1))
# np.savetxt(os.path.join(SAVE_ROOT, f"cam_y.txt"),
# np.concatenate([center, y_axis], axis=-1))
# np.savetxt(os.path.join(SAVE_ROOT, f"cam_x.txt"),
# np.concatenate([center, x_axis], axis=-1))

# Flip axes to match convention
c2w_matrices[:, :3, 2] = c2w_matrices[:, :3, 2] * (-1)
c2w_matrices[:, :3, 1] = c2w_matrices[:, :3, 1] * (-1)

# Assume image filenames are “000000_0000.png”, “000000_0001.png”, ..., “000000_0020.png”
image_names = [f"000000_{i:04d}.png" for i in range(0, 21)]

# Assume camera ID = 1 (if each image has a different camera ID, modify accordingly)
camera_id = 1

image_lines = []

c2w_new = []
for i, c2w in enumerate(c2w_matrices):
    # Invert the matrix and apply axis transformations
    w2c = np.linalg.inv(c2w) @ np.array([[1, 0, 0, 0],
                                         [0, 0, -1, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1]], dtype=np.float) \
                           @ np.array([[0, 0, -1, 0],
                                       [0, 1, 0, 0],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 1]], dtype=np.float).T

    # Extract rotation and translation
    R_matrix = w2c[:3, :3]
    t = w2c[:3, 3]

    print('R', R_matrix)
    print('t', t)

    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(R_matrix)
    quaternion = rotation.as_quat()
    qw, qx, qy, qz = quaternion[3], quaternion[0], quaternion[1], quaternion[2]

    # Generate image ID (from 1 to 21)
    image_id = i + 1

    # Generate line in COLMAP format
    result_line = f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {camera_id} {image_names[i]}\n"
    image_lines.append(result_line)

# Alternative transformation option (commented out)
# c2w_tmp = c2w_matrices.copy()
# c2w_matrices[:, :3, 1] = c2w_tmp[:, :3, 2]
# c2w_matrices[:, :3, 2] = c2w_tmp[:, :3, 1]
# c2w_matrices[:, 1, 3] = c2w_tmp[:, 2, 3]
# c2w_matrices[:, 2, 3] = c2w_tmp[:, 1, 3]
# cam_poses = c2w_new
# poses = np.array(cam_poses)
# x_axis = poses[:, :3, 0]
# y_axis = poses[:, :3, 1]
# z_axis = poses[:, :3, 2]
# center = poses[:, :3, 3]
# SAVE_ROOT = '.'
# np.savetxt(os.path.join(SAVE_ROOT, f"cam_z.txt"),
# np.concatenate([center, z_axis], axis=-1))
# np.savetxt(os.path.join(SAVE_ROOT, f"cam_y.txt"),
# np.concatenate([center, y_axis], axis=-1))
# np.savetxt(os.path.join(SAVE_ROOT, f"cam_x.txt"),
# np.concatenate([center, x_axis], axis=-1))

# Write all lines to images.txt file
with open("images.txt", "w") as file:
    file.writelines(image_lines)

print("Done writing images.txt")

