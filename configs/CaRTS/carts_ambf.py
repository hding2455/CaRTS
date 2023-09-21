from torch.optim import  SGD, Adam
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss, SmoothL1Loss
from torch.optim.lr_scheduler import StepLR
import numpy as np

path_to_this_repo = "/home/hao/CaRTS/"
mesh_path = path_to_this_repo + "ADF/low_res/"
link_meshes = [mesh_path+"base link.obj", mesh_path+"yaw link.obj", mesh_path+"pitch end link.obj",
               mesh_path+"main insertion link.obj", mesh_path+"tool roll link.obj", mesh_path+"tool pitch link.obj",
              mesh_path+"tool yaw link.obj"]
tool_meshes = [mesh_path+"tool gripper1 link.obj", mesh_path+"tool gripper2 link.obj"]
baseT1 = [[-8.6602e-01, 3.2138e-01, 3.8303e-01,1.0],
          [-2.1955e-06, -7.6607e-01, 6.4276e-01,1.3461],
          [5.0000e-01, 5.5665e-01,  6.6343e-01,1.3499],
          [        0.0,         0.0,         0.0,1.0]]
baseT2 = [[-8.6602e-01, -3.2138e-01, -3.8303e-01, -1.0],
        [-2.1955e-06, -7.6607e-01,  6.4276e-01, 1.3461],
        [-5.0000e-01,  5.5665e-01,  6.6343e-01, 1.3499],
         [        0.0,         0.0,         0.0,1.0]]
link_DH2Mesh = [
                    [[-1.0, 0.0, 0.0,  0.0],
                      [ 0.0, 0.0, 1.0, 0.0],
                      [ 0.0, 1.0, 0.0, 0.0],
                      [ 0.0, 0.0, 0.0, 1.0]],
                    [[ -1.0, 0.0, 0.0,  0.43],
                      [ 0.0, -1.0, 0.0, 1.441],
                      [ 0.0, 0.0, 1.0, 0.0],
                      [ 0.0, 0.0, 0.0, 1.0]],
                    [[ 0.0, -1.0, 0.0,  0],
                      [ 0.0, 0.0,-1.0, 0],
                      [1.0, 0.0, 0.0, 0],
                      [ 0.0, 0.0, 0.0, 1.0]],
                    [[ 1.0, 0.0, 0.0,  0.0],
                      [ 0.0, 1.0, 0.0, 0.0],
                      [ 0.0, 0.0, 1.0, -1.85],
                      [ 0.0, 0.0, 0.0, 1.0]],
                    [[ 1.0, 0.0, 0.0,  0.0],
                      [ 0.0, 1.0, 0.0, 0.0],
                      [ 0.0, 0.0, 1.0, 0.0],
                      [ 0.0, 0.0, 0.0, 1.0]],
                    [[ 1.0, 0.0, 0.0,  0.0],
                      [ 0.0, -1.0, 0.0, 0.00],
                      [ 0.0, 0.0, -1.0, 0.0],
                      [ 0.0, 0.0, 0.0, 1.0]],
                    [[ 1.0, 0.0, 0.0,  0.0],
                      [ 0.0, 1.0, 0.0,  0.0],
                      [ 0.0, 0.0, 1.0,  0.0],
                      [ 0.0, 0.0, 0.0, 1.0]]
        ]
tool_DH2Mesh = [
                     [[ 0.0, 0.0, -1.0,  0.0],
                      [ -1.0, 0.0, 0.0,  0.0],
                      [ 0.0, 1.0, 0.0,  -0.106],
                      [ 0.0, 0.0, 0.0, 1.0]],
                    [[ 0.0, 0.0, -1.0,  0.0],
                      [ -1.0, 0.0, 0.0,  0.0],
                      [ 0.0, 1.0, 0.0,  -0.106],
                      [ 0.0, 0.0, 0.0, 1.0]]
        ]

class cfg:
    train_dataset = dict(
        name = "AMBFSim",
        args = dict(
            series_length = 1,
            folder_path = "/data/hao/new_ambf_dataset",
            video_paths = ["Video_01", "Video_02", "Video_03", "Video_04", "Video_05", "Video_06",
                           "synthetics-Video_01", "synthetics-Video_02", "synthetics-Video_03", "synthetics-Video_04",
                           "synthetics-Video_05", "synthetics-Video_06"]))
    validation_dataset = dict(
        name = "AMBFSim",
        args = dict(
            series_length = 1,
            folder_path = "/data/hao/new_ambf_dataset",
            video_paths = ["Video_08"]))
    model = dict(
        name = "CaRTS",
        params = dict(
            vision = dict(
                name = "Unet",
                params = dict(
                    input_dim = 3,
                    hidden_dims = [512, 256, 128, 64, 32],
                    size = (15, 20),
                    target_size = (480, 640),
                    criterion = BCELoss(),
                    train_params = dict(
                        lr_scheduler = dict(
                            lr_scheduler_class = StepLR,
                            args = dict(
                                step_size=20,
                                gamma=0.1)),
                        optimizer = dict(
                            optim_class = SGD,
                            args = dict(
                                lr = 0.01,
                                momentum = 0.9,
                                weight_decay = 10e-5)),
                        max_epoch_number=50,
                        save_interval=10,
                        save_path='./checkpoints/carts_base_ambf_unet/',
                        log_interval=50))),
            optim = dict(
                name = "AttFeatureCosSimOptim",
                params = dict(
                    optimizer = dict(
                        optim_class = Adam,
                        args = dict(
                            lr = 1e-3)),
                    lr_scheduler = dict(
                        lr_scheduler_class = StepLR,
                        args = dict(
                        step_size=5,
                        gamma=0.9)),
                    loss_function = dict(
                        loss_function_class = SmoothL1Loss,
                        args = dict(beta=1.0,
                                reduction='mean')),
                    iteration_num = 50,
                    background_image = '/data/hao/new_ambf_dataset/background.png',
                    grad_limit = 1.0,
                    optimize_cameras=False,
                    optimize_kinematics=True,
                    ref_threshold=0.5,
                    render = dict(name = "BaseRender",
                        params = dict(
                            camera_params = {'name':"FoVPerspectiveCameras",
                                             'args':dict(znear=0.01, zfar=10.0, fov=1.2/3.1416*180),
                                             'position':[0.02,  6.4169e-01,  1.2882e+00],
                                             'at':[0.0,  3.2031e-01,  9.0520e-01],
                                             'up':[0.0, -7.6606e-01,  6.4276e-01]},
                            robot_params_list = [{'name': "PSM",
                                                  'args': dict(link_meshes = link_meshes,
                                                               tool_meshes = tool_meshes,
                                                               baseT = baseT1,
                                                               link_DH2Mesh = link_DH2Mesh,
                                                               tool_DH2Mesh = tool_DH2Mesh)
                                                 },
                                                 {'name': "PSM",
                                                  'args': dict(link_meshes = link_meshes,
                                                               tool_meshes = tool_meshes,
                                                               baseT = baseT2,
                                                               link_DH2Mesh = link_DH2Mesh,
                                                               tool_DH2Mesh = tool_DH2Mesh)
                                                 }],
                            render_params = {'blend_params':dict(sigma=1e-4, gamma=1e-4),
                                             'raster_settings_silhouette':dict(image_size=(480,640), blur_radius=np.log(1. / 1e-4 - 1.) * 1e-4, faces_per_pixel=150),
                                             'raster_settings_image':dict(image_size=(480,640), blur_radius=0, faces_per_pixel=1),
                                             'lights':dict(name='PointLights', args=dict(location=((0.0, 6.4169e-01, 1.2882e+00),)))}))))))
