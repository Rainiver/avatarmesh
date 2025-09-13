import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform

# 加载粗糙的 mesh
coarse_mesh_path = '/data/vde/zhongyuhe/workshop/InstantMesh/outputs/instant-mesh-large/meshes/000000_1.obj'
coarse_mesh = load_objs_as_meshes(coarse_mesh_path, device="cuda")

index = randint(0, len(viewpoint_stack) - 1)
# print('index', index)
# viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
viewpoint_cam = viewpoint_stack.pop(index)
# 找到viewpoint_cam的序号，然后load 对应的mesh

render_pkg = render(viewpoint_cam, gaussians, pipe, background)
image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
render_pkg["visibility_filter"], render_pkg["radii"]
# print('image.shape', image.shape)
gt_image = viewpoint_cam.original_image.cuda()
# print('gt_image.shape', gt_image.shape)

# 引入伪GT的normal
pre_obj_path = f"/data/vdd/zhongyuhe/workshop/SIFU-main/results_test_full/sifu/obj/000000_{index:04d}_refine.obj"
meshes = load_objs_as_meshes([obj_path])
raster_settings = RasterizationSettings(
    image_size=576,
    blur_radius=0.0,
    faces_per_pixel=1,
)
rasterizer = MeshRasterizer(cameras=viewpoint_cam, raster_settings=raster_settings)
fragments = rasterizer(meshes)
normals = meshes.verts_normals_packed()

# 从投影片段中获取对应的三角面片，并计算法向量转换到图像平面
pix_normals = torch.zeros((576, 576, 3), device='cuda')
for i in range(576):
    for j in range(576):
        face_idx = fragments.pix_to_face[0, i, j]
        if face_idx >= 0:
            pix_normals[i, j, :] = normals[meshes.faces_packed()[face_idx]]
# 归一化pix_normals到[0, 1]
pix_normals_normalized = (pix_normals + 1.0) / 2.0

# 转换为numpy数组并映射到[0, 255]
pix_normals_uint8 = (pix_normals_normalized * 255).cpu().numpy().astype(np.uint8)


# 加载相机参数 (假设使用 colmap 格式)
def load_camera_params():
    # 具体实现需根据相机参数格式
    cameras = []
    for i in range(21):
        R, T = look_at_view_transform(dist=2.7, elev=10, azim=i*18)  # 假设了一些参数
        # 可根据实际相机参数进行调整
        cameras.append(PerspectiveCameras(device="cuda", R=R, T=T))
    return cameras

cameras = load_camera_params()

from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import rasterize_meshes
import torch.nn.functional as F

def compute_normals(mesh):
    verts = mesh.verts_packed()  # 获取所有顶点
    faces = mesh.faces_packed()  # 获取所有面
    face_normals = mesh.faces_normals_packed()  # 获取所有面的法向量
    return verts, faces, face_normals

verts, faces, face_normals = compute_normals(mesh)

def render_normals(mesh, cameras):
    # 使用光栅化（rasterization）方法生成法向量图
    raster_settings = RasterizationSettings(
        image_size=512,  # 图像尺寸
        blur_radius=0.0,  # 模糊半径
        faces_per_pixel=1  # 每个像素的最大面数
    )

    fragments = rasterize_meshes(mesh, cameras=cameras, raster_settings=raster_settings)
    return fragments

# 在所有视角下计算法向量图
all_fragments = [render_normals(mesh, cam) for cam in cameras]

from pytorch3d.ops import sample_points_from_meshes
from torch.optim import Adam


# 假设已知法相贴图 gt_normals_list 是一个 21 个元素的列表，每个元素是 (H, W, 3) 形状的 tensor
def compute_loss(mesh, all_fragments, gt_normals_list):
    loss = 0
    for i, fragments in enumerate(all_fragments):
        # 获取当前视角的法向量图
        # face_indices = fragments.pix_to_face[0]
        vert_indices = fragments.faces_uvs_idx
        pixel_normals = torch.ones_like(gt_normals_list[i])

        # 将法向量与真实法相贴图进行比对
        gt_normals = gt_normals_list[i].to(pixel_normals.device)
        mask = (gt_normals.sum(dim=-1) != 0)
        # 使用均方误差
        loss += F.mse_loss(pixel_normals[mask], gt_normals[mask])

    return loss


optimizer = Adam(mesh.verts_packed(), lr=0.01)

# 获取真实法相贴图
gt_normals_list = [load_gt_normal_map(i) for i in range(21)]  # 假设存在加载函数 load_gt_normal_map

for epoch in range(num_epochs):
    optimizer.zero_grad()
    all_fragments = [render_normals(mesh, cam) for cam in cameras]
    loss = compute_loss(mesh, all_fragments, gt_normals_list)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
