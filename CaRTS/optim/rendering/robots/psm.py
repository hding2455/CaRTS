import numpy as np
import torch
import torch.nn as nn
import pytorch3d
from pytorch3d.io import load_obj
from .DH import *
PI = np.pi
PI_2 = np.pi/2

class PSMJointMapping:
    def __init__(self):
        self.idx_to_name = {0: 'baselink-yawlink',
                            1: 'yawlink-pitchbacklink',
                            2: 'pitchendlink-maininsertionlink',
                            3: 'maininsertionlink-toolrolllink',
                            4: 'toolrolllink-toolpitchlink',
                            5: 'toolpitchlink-toolyawlink'}

        self.name_to_idx = {'baselink-yawlink': 0,
                            'yawlink-pitchbacklink': 1,
                            'pitchendlink-maininsertionlink': 2,
                            'maininsertionlink-toolrolllink': 3,
                            'toolrolllink-toolpitchlink': 4,
                            'toolpitchlink-toolyawlink': 5}

class PSM(nn.Module):
    def __init__(self, link_meshes, tool_meshes, baseT, link_DH2Mesh, tool_DH2Mesh, device,
                 L_rcc = 4.389,  # From dVRK documentation x 10
                 L_tool = 4.16,  # From dVRK documentation x 10
                 L_pitch2yaw = 0.091,  # Fixed length from the palm joint to the pitch joint
                 L_yaw2ctrlpnt = 0.106):
        #robot parameters
        super().__init__()
        self.num_links = 7
        self.num_tools = 2
        self.L_rcc = L_rcc  # From dVRK documentation x 10
        self.L_tool = L_tool  # From dVRK documentation x 10
        self.L_pitch2yaw = L_pitch2yaw  # Fixed length from the palm joint to the pitch joint
        self.L_yaw2ctrlpnt = L_yaw2ctrlpnt  # Fixed length from the pinch joint to the pinch tip
        # Delta between tool tip and the Remote Center of Motion
        self.scale = 1.0
        self.baseT = nn.Parameter(torch.tensor(baseT).to(device=device, dtype=torch.float32))
        
        self.tool_joint_limits = PI_2

        # PSM DH Params from link 1-7
        # alpha | a | theta | d | offset | type
        self.kinematics = [DH(PI_2, 0, 0, 0, PI_2, 
                              JointType.REVOLUTE, Convention.MODIFIED, device),
                           DH(-PI_2, 0, 0, 0, -PI_2,
                              JointType.REVOLUTE, Convention.MODIFIED, device),
                           DH(PI_2, 0, 0, 0, -self.L_rcc,
                              JointType.PRISMATIC, Convention.MODIFIED, device),
                           DH(0, 0, 0, self.L_tool, 0,
                              JointType.REVOLUTE, Convention.MODIFIED, device),
                           DH(-PI_2, 0, 0, 0, -PI_2,
                              JointType.REVOLUTE, Convention.MODIFIED, device),
                           DH(-PI_2, self.L_pitch2yaw, 0, 0, -PI_2,
                              JointType.REVOLUTE, Convention.MODIFIED, device),
#                            DH(-PI_2, 0, 0, 0, PI_2, 
#                               JointType.REVOLUTE, Convention.MODIFIED, device)]
                           DH(-PI_2, 0, 0, self.L_yaw2ctrlpnt, PI_2, 
                              JointType.REVOLUTE, Convention.MODIFIED, device)]
        
        # PSM DH params for tools
        self.tools_kinematics = [DH(0, 0, 0, 0, 0, 
                              JointType.REVOLUTE, Convention.MODIFIED, device),
                                DH(0, 0, 0, 0, 0, 
                              JointType.REVOLUTE, Convention.MODIFIED, device)]
        
        self.link_DH2Mesh = torch.tensor(link_DH2Mesh).to(device=device, dtype=torch.float32)
        
        self.tool_DH2Mesh = torch.tensor(tool_DH2Mesh).to(device=device, dtype=torch.float32)
        

        #load meshes 
        self.vertices = []
        self.face_ids = []
        self.atlas = []

        self.device = device
        for mesh in link_meshes:
            verts, faces, aux = load_obj(mesh, create_texture_atlas = True)
            self.vertices.append(verts.to(device=device, dtype=torch.float32))
            self.face_ids.append(faces.verts_idx.to(device=device, dtype=torch.float32))
            self.atlas.append(aux.texture_atlas.to(device=device, dtype=torch.float32))
        
        for mesh in tool_meshes:
            verts, faces, aux = load_obj(mesh, create_texture_atlas = True)
            self.vertices.append(verts.to(device=device, dtype=torch.float32))
            self.face_ids.append(faces.verts_idx.to(device=device, dtype=torch.float32))
            self.atlas.append(aux.texture_atlas.to(device=device, dtype=torch.float32))
            
    def fk(self, joint_pos):
        verts = []
        face_ids = []
        atlases = []
        verts_off = 0
                           
        j = torch.tensor([0, 0, 0, 0, 0, 0, 0]).to(device=self.device, dtype=torch.float32)
        for i in range(self.num_links-1):
            j[i] = joint_pos[i]
        tool_angle = (joint_pos[6]) * self.tool_joint_limits
        tool_angles = torch.stack([-tool_angle, tool_angle])
        T_N_0 = torch.tensor(np.identity(4)).to(device=self.device, dtype=torch.float32)
        Trans = torch.matmul(self.baseT, T_N_0)
        for i in range(self.num_links):
            R = Trans[:3,:3]
            T = Trans[:3, 3]
            scale = self.scale
            t = pytorch3d.transforms.transform3d.Transform3d().scale(scale).rotate(R.T).translate(*T).to(device=self.device)
            vert = t.transform_points(self.vertices[i])
            verts.append(vert)
            face_ids.append(self.face_ids[i]+verts_off)
            atlases.append(self.atlas[i])
            verts_off += len(vert)
            
            link_dh = self.kinematics[i]

            T_N_0 = torch.matmul(T_N_0 , link_dh.mat_from_dh(j[i]))

            Trans = torch.matmul(self.baseT, torch.matmul(T_N_0 , self.link_DH2Mesh[i]))
        
        self.DH = T_N_0
            
        for i in range(self.num_tools):
            tool_dh = self.tools_kinematics[i]

            Trans = torch.matmul(torch.matmul(torch.matmul(self.baseT,T_N_0), self.tool_DH2Mesh[i]),  tool_dh.mat_from_dh(tool_angles[i]))
            R = Trans[:3,:3]
            T = Trans[:3, 3]
            scale = self.scale
            t = pytorch3d.transforms.transform3d.Transform3d().scale(scale).rotate(R.T).translate(*T).to(device=self.device)
            vert = t.transform_points(self.vertices[i+self.num_links])
            verts.append(vert)
            face_ids.append(self.face_ids[i+self.num_links]+verts_off)
            atlases.append(self.atlas[i+self.num_links])
            verts_off += len(vert)

        atlas = torch.cat(atlases)
        verts = torch.cat(verts)
        faces = torch.cat(face_ids)

        return verts, faces, atlas

