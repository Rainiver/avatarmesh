import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform

# Load coarse mesh
coarse_mesh_path = '/data/vde/zhongyuhe/workshop/InstantMesh/outputs/instant-mesh-large/meshes/000000_1.obj'
coarse_mesh = load_objs_as_meshes(coarse_mesh_path, device="cuda")

index = randint(0, len(viewpoint_stack) - 1)
# print('index', index)
# viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
viewpoint_cam = viewpoint_stack.pop(index)
# Find the index of viewpoint_cam and then load the corresponding mesh

render_pkg = render(viewpoint_cam, gaussians, pipe, background)
image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
render_pkg["visibility_filter"], render_pkg["radii"]
# print('image.shape', image.shape)
gt_image = viewpoint_cam.original_image.cuda()
# print('gt_image.shape', gt_image.shape)

# Load pseudo-GT normals
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

# Extract triangle faces from rasterized fragments and convert normals to image space
pix_normals = torch.zeros((576, 576, 3), device='cuda')
for i in range(576):
    for j in range(576):
        face_idx = fragments.pix_to_face[0, i, j]
        if face_idx >= 0:
            pix_normals[i, j, :] = normals[meshes.faces_packed()[face_idx]]
# Normalize pix_normals to [0, 1]
pix_normals_normalized = (pix_normals + 1.0) / 2.0

# Convert to numpy array and map to [0, 255]
pix_normals_uint8 = (pix_normals_normalized * 255).cpu().numpy().astype(np.uint8)


# Load camera parameters (assume colmap format)
def load_camera_params():
    # Implementation depends on actual camera parameter format
    cameras = []
    for i in range(21):
        R, T = look_at_view_transform(dist=2.7, elev=10, azim=i*18)  # Example parameters
        # Can be adjusted based on real camera parameters
        cameras.append(PerspectiveCameras(device="cuda", R=R, T=T))
    return cameras

cameras = load_camera_params()

from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import rasterize_meshes
import torch.nn.functional as F

def compute_normals(mesh):
    verts = mesh.verts_packed()  # All vertices
    faces = mesh.faces_packed()  # All faces
    face_normals = mesh.faces_normals_packed()  # Normal vector of each face
    return verts, faces, face_normals

verts, faces, face_normals = compute_normals(mesh)

def render_normals(mesh, cameras):
    # Generate normal maps using rasterization
    raster_settings = RasterizationSettings(
        image_size=512,  # Image size
        blur_radius=0.0,  # Blur radius
        faces_per_pixel=1  # Max number of faces per pixel
    )

    fragments = rasterize_meshes(mesh, cameras=cameras, raster_settings=raster_settings)
    return fragments

# Compute normal maps for all viewpoints
all_fragments = [render_normals(mesh, cam) for cam in cameras]

from pytorch3d.ops import sample_points_from_meshes
from torch.optim import Adam


# Assume gt_normals_list is a list of 21 elements, each (H, W, 3) tensor representing the ground-truth normal map
def compute_loss(mesh, all_fragments, gt_normals_list):
    loss = 0
    for i, fragments in enumerate(all_fragments):
        # Get normal map of current viewpoint
        # face_indices = fragments.pix_to_face[0]
        vert_indices = fragments.faces_uvs_idx
        pixel_normals = torch.ones_like(gt_normals_list[i])

        # Compare predicted normals with GT normal map
        gt_normals = gt_normals_list[i].to(pixel_normals.device)
        mask = (gt_normals.sum(dim=-1) != 0)
        # Use MSE loss
        loss += F.mse_loss(pixel_normals[mask], gt_normals[mask])

    return loss


optimizer = Adam(mesh.verts_packed(), lr=0.01)

# Load ground-truth normal maps
gt_normals_list = [load_gt_normal_map(i) for i in range(21)]  # Assume function load_gt_normal_map exists

for epoch in range(num_epochs):
    optimizer.zero_grad()
    all_fragments = [render_normals(mesh, cam) for cam in cameras]
    loss = compute_loss(mesh, all_fragments, gt_normals_list)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

