from torch.optim import  SGD, Adam
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss, SmoothL1Loss
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
path_to_this_repo = "/home/hao/CausalMedSeg/"
mesh_path = path_to_this_repo + "ADF/low_res/"
link_meshes = [mesh_path+"base link.obj", mesh_path+"yaw link.obj", mesh_path+"pitch end link.obj",
               mesh_path+"main insertion link.obj", mesh_path+"tool roll link.obj", mesh_path+"tool pitch link.obj",
              mesh_path+"tool yaw link.obj"]
tool_meshes = [mesh_path+"tool gripper1 link.obj", mesh_path+"tool gripper2 link.obj"]
baseT1 = [[ 0.7886,  0.4085, -0.4194, -1.4415],
        [ 0.4739, -0.8599,  0.2190,  0.2855],
        [-0.2747, -0.3400, -0.9099,  0.6532],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
baseT2 = [[ 0.5483, -0.8265, -0.1314,  1.0903],
        [-0.7020, -0.5704,  0.4681, -0.5071],
        [-0.4500, -0.1565, -0.8751,  0.0092],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
#baseT1 = [[ 0.8258,  0.4259, -0.3523, -1.6137],
#        [ 0.5095, -0.8261,  0.1959,  0.0019],
#        [-0.2045, -0.3428, -0.9196,  0.4702],
#        [ 0.0000,  0.0000,  0.0000,  1.0000]]
#baseT2 = [[ 0.5445, -0.8148, -0.1687,  1.1866],
#        [-0.6761, -0.5646,  0.4704, -0.4388],
#        [-0.4720, -0.1431, -0.8683,  0.1136],
#        [ 0.0000,  0.0000,  0.0000,  1.0000]]

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
        name = "CausalToolSeg",
        args = dict(
            series_length = 1,
            folder_path = "/data/hao/processed_data",
            video_paths = ["set-1", "set-2", "set-3", "set-5", "set-6", "set-9", "set-10", 
                          'synthetics-set-1',  'synthetics-set-2' , 'synthetics-set-3' , 'synthetics-set-5',  
                          'synthetics-set-6',  'synthetics-set-9', 'synthetics-set-10'],
            subset_paths = ["regular"]))
    validation_dataset = dict(
        name = "CausalToolSeg",
        args = dict(
            series_length = 1,
            folder_path = "/data/hao/processed_data",
            video_paths = ["set-11"],
            subset_paths = ["blood"]))
    carts = dict(
        name = "CaRTSBase",
        params = dict(
            vision = dict(
                name = "Unet",
                params = dict(
                    input_dim = 3,
                    hidden_dims = [512, 256, 128, 64, 32],
                    size = (15, 20),
                    target_size = (360, 500),
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
                        save_path='./checkpoints/carts_base_cts/',
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
                    background_image = '/data/hao/processed_data/mean_background_l.png',
                    grad_limit = 1.0,
                    iteration_num = 30,
                    optimize_cameras=False,
                    optimize_kinematics=True,
                    ref_threshold=0.5,
                    render = dict(name = "BaseRender",
                        params = dict(
                            camera_params = {'name':"FoVPerspectiveCameras", 
                                'args':dict(K = torch.tensor([[[ 4.6714,  0.0000,  0,  0.0000],
                                                               [ 0.0000,  4.6542,  0,  0.0000],
                                                               [ 1.5856,  1.31000,  1.0001,  -0.010000],
                                                               [ 0.0000,  0.0000, 1.0000,  0.0000]]])),
                                            'position':[-0.02,  0.05,  -0.3],
                                            'at':[-0.02,  0.05,  0.0],
                                            'up':[0.0, 1.0,  0.0]},
                            robot_params_list = [{'name': "PSM",
                                                  'args': dict(link_meshes = link_meshes,
                                                               tool_meshes = tool_meshes,
                                                               baseT = baseT1,
                                                               link_DH2Mesh = link_DH2Mesh,
                                                               tool_DH2Mesh = tool_DH2Mesh,
                                                               L_rcc = 4.318)
                                                 },
                                                 {'name': "PSM",
                                                  'args': dict(link_meshes = link_meshes,
                                                               tool_meshes = tool_meshes,
                                                               baseT = baseT2,
                                                               link_DH2Mesh = link_DH2Mesh,
                                                               tool_DH2Mesh = tool_DH2Mesh,
                                                               L_rcc = 4.318)
                                                 }],
                            render_params = {'blend_params':dict(sigma=1e-4, gamma=1e-4),
                                             'raster_settings_silhouette':dict(image_size=(360,480), blur_radius=0, faces_per_pixel=150),
                                             'raster_settings_image':dict(image_size=(360,480), blur_radius=0, faces_per_pixel=1),
                                             'lights':dict(name='PointLights', args=dict(location=((0.0, 0.0, 0.0),)))}))))))
