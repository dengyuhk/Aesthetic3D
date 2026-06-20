import torch
import numpy as np
import math
# from pytorch3d.structures import Meshes
# from pytorch3d.io import load_objs_as_meshes, load_obj
# from pytorch3d.renderer import (
#     Textures,
#     look_at_view_transform,
#     FoVPerspectiveCameras,
#     MeshRenderer,
#     MeshRasterizer,
#     SoftSilhouetteShader,
#     RasterizationSettings,
#     PointLights,
#     DirectionalLights,
#     Materials,
#     SoftPhongShader,
#     HardFlatShader,
#     TexturesVertex
# )
import trimesh
import pyrender
# import pyredner
# import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# device = "cuda" if torch.cuda.is_available() else "cpu"

# render_single_mesh_1, render_single_mesh_2, render_single_mesh_3 use 3 different rendering library.
# for all the renderer, verts should be normalized to diagonal 1
# finally render_single_mesh_3 is used


# def render_single_mesh_1(verts, faces, num_views, elev=15, resolution=128):
#     res = resolution
#     # Initialize each vertex to be white in color.
#     verts = verts.to(device)
#     faces = faces.to(device)
#     verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
#     textures = TexturesVertex(verts_features=verts_rgb.to(device))
#     mesh = Meshes(
#         verts=[verts],
#         faces=[faces],
#         textures=textures
#     )

#     meshes = mesh.extend(num_views)

#     # Get a batch of viewing angles.
#     # elev = 15
#     azim = torch.linspace(-180, 180, num_views + 1)[:num_views]

#     R, T = look_at_view_transform(dist=1, elev=elev, azim=azim)
#     cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

#     dirs = cameras.get_camera_center() / torch.linalg.norm(cameras.get_camera_center(), dim=1, keepdim=True)
#     lights = DirectionalLights(device=device, direction=dirs, specular_color=torch.zeros_like(dirs).to(device))

#     sigma = 1e-4

#     raster_settings = RasterizationSettings(
#         image_size=res,
#         blur_radius=0,
#         faces_per_pixel=1,
#         z_clip_value=0.5,
#         # cull_backfaces=True
#     )

#     raster_settings_soft = RasterizationSettings(
#         image_size=res,
#         blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
#         faces_per_pixel=10,
#         perspective_correct=False,
#     )

#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(
#             cameras=cameras,
#             raster_settings=raster_settings,
#         ),
#         shader=HardFlatShader(device=device,
#                               cameras=cameras,
#                               lights=lights)
#     )
#     return renderer(meshes)


# def render_single_mesh_2(verts, faces, num_views, resolution=128, ambient=0.5, diff=0.3):
#     res = resolution
#     dist = 1
#     elev = math.pi / 12
#     camera0 = pyredner.Camera(position=torch.tensor([0.0, dist * math.sin(elev), - dist * math.cos(elev)]), look_at=torch.tensor([0.0, 0.0, 0]), up=torch.tensor([0., 1., 0.]), fov=torch.tensor([60.]), resolution=(res, res))
#     cameras = [camera0]
#     for i in range(1, num_views):
#         rot = pyredner.gen_rotate_matrix(torch.Tensor([0., 2 * math.pi / num_views * i, 0.]))
#         cameras.append(pyredner.Camera(position=(camera0.position @ rot).view(-1), look_at=camera0.look_at, up=camera0.up,
#                                        fov=camera0.fov, resolution=camera0.resolution))
#     pyredner.render_pytorch.print_timing = False

#     diffuse = math.pi * diff
# #     diffuse = 1.
#     material = pyredner.Material(diffuse_reflectance=torch.tensor((diffuse, diffuse, diffuse), device=pyredner.get_device()))
#     objects = pyredner.Object(vertices=verts, indices=faces.int(), material=material)

#     lights = []
#     scenes = []
#     for i in range(num_views):
#         dirs = -cameras[i].position / torch.linalg.norm(cameras[i].position)
#         # DirectionalLight in render_utils.py (in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/pyredner) has been changed to 'double rendering'
#         light = pyredner.DirectionalLight(dirs, intensity=torch.tensor([1.0, 1.0, 1.0], device=pyredner.get_device()))
#         lights.append([light])
#         scenes.append(pyredner.Scene(camera=cameras[i], objects=[objects]))
#     imgs = pyredner.render_deferred(scene=scenes, lights=lights, alpha=True)
#     imgs[:, :, :, :3] += ambient * imgs[:, :, :, 3:4]
#     return imgs

def RotX(theta):
    return np.array([[1., 0., 0.], [0., math.cos(theta), -math.sin(theta)], [0., math.sin(theta), math.cos(theta)]])


def RotY(theta):
    return np.array([[math.cos(theta), 0, math.sin(theta)], [0., 1., 0.], [-math.sin(theta), 0, math.cos(theta)]])


def render_single_mesh_3(verts, faces, num_views=12, resolution=256):
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[.0, .0, .0])
    material = pyrender.MetallicRoughnessMaterial(alphaMode='BLEND', baseColorFactor=[0.3, 0.3, 0.3, 1.0], metallicFactor=0.0, roughnessFactor=1.0, doubleSided=False)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, material=material)
    dist = 1.0
    elev = math.pi / 12
    camera_pose = np.eye(4)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1500)
    camera_node = scene.add(camera)
    light_node = scene.add(light)
    scene.add(mesh)
    r = pyrender.OffscreenRenderer(resolution, resolution)
    imgs = []
    for i in range(num_views):
        azim = 2 * math.pi / num_views * i
        camera_pose[:3, :3] = RotY(azim) @ RotX(-elev)
        camera_pose[:3, 3] = np.array([math.cos(elev) * math.sin(azim), math.sin(elev), math.cos(elev) * math.cos(azim)])
        scene.set_pose(camera_node, pose=camera_pose)
        scene.set_pose(light_node, pose=camera_pose)
        color, _ = r.render(scene)
        color = color / 255.0
        imgs.append(color[:, :, :])
    imgs = np.stack(imgs)
    return imgs
