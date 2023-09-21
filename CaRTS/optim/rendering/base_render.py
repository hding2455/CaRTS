
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from .robots import robots_dict
# io utils
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex, TexturesAtlas
)

camera_dict = {"FoVPerspectiveCameras":FoVPerspectiveCameras}
lights_dict = {"PointLights":PointLights}

class BaseRender(nn.Module):

    def __init__(self, params, device):
        super().__init__()
        camera_params = params['camera_params']
        robot_params_list = params['robot_params_list']
        render_params = params['render_params']
        self.device = device
        #init camera
        self.camera = camera_dict[camera_params['name']](**(camera_params['args']), device=device)
        self.camera_position = nn.Parameter(torch.tensor(camera_params['position'])).to(device = device)
        self.camera_at = nn.Parameter(torch.tensor(camera_params['at'])).to(device = device)
        self.camera_up = nn.Parameter(torch.tensor(camera_params['up'])).to(device = device)

        #init robots
        self.robots = []
        for robot_params in robot_params_list:
            robot = robots_dict[robot_params['name']](**(robot_params['args']), device=device)
            self.robots.append(robot)

        self.blend_params = BlendParams(**render_params['blend_params'])
        self.raster_settings_silhouette = RasterizationSettings(**render_params['raster_settings_silhouette'])
        self.silhouette_renderer = MeshRenderer(
                                 rasterizer=MeshRasterizer(
                                 cameras=self.camera,
                                 raster_settings=self.raster_settings_silhouette),
                                 shader=SoftSilhouetteShader(blend_params=self.blend_params))
        self.raster_settings_image = RasterizationSettings(**render_params['raster_settings_image'])
        self.lights = lights_dict[render_params['lights']['name']](device=self.device,**(render_params['lights']['args']))
        self.phong_renderer = MeshRenderer(
                            rasterizer=MeshRasterizer(
                            cameras=self.camera,
                            raster_settings=self.raster_settings_image),
                            shader=HardPhongShader(device=device, cameras=self.camera, lights=self.lights))

    def forward(self, kinematics):
        mesh = self.get_mesh(kinematics)
        R = look_at_rotation(self.camera_position[None, :],at=self.camera_at[None, :],up=self.camera_up[None, :],device=self.device)
        T = -torch.bmm(R.transpose(2, 1), self.camera_position[None, :, None])[:, :, 0]
        #silhouette = self.silhouette_renderer(meshes_world=mesh, R=R, T=T)
        image = self.phong_renderer(meshes_world=mesh, R=R, T=T)
        return image, image
    
    def get_mesh(self, kinematics):
        verts = []
        face_ids = []
        atlases = []
        verts_off = 0
        for i in range(len(self.robots)):
            vert, face_id, atlas = self.robots[i].fk(kinematics[i])
            verts.append(vert)
            face_ids.append(face_id+verts_off)
            atlases.append(atlas)
            verts_off += len(vert)
        atlas = torch.cat(atlases)
        verts = torch.cat(verts)
        faces = torch.cat(face_ids)
        tex = TexturesAtlas(atlas=[atlas])
        mesh = Meshes(
            verts=[verts], faces=[faces], textures=tex
        )
        return mesh
