import numpy as np
import torch
from enum import Enum

PI = 3.1415926
PI_2 = PI/2

class JointType(Enum):
    REVOLUTE = 0
    PRISMATIC = 1


class Convention(Enum):
    STANDARD = 0
    MODIFIED = 1

class DH:
    def __init__(self, alpha, a, theta, d, offset, joint_type, convention, device):
        self.alpha = alpha
        self.a = a
        self.theta = theta
        self.d = d
        self.offset = offset
        self.joint_type = joint_type
        self.convention = convention
        self.device = device

    def mat_from_dh(self, theta):
        alpha = self.alpha
        a = self.a
        d = self.d 
        offset = self.offset
        joint_type = self.joint_type
        convention = self.convention
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        th = 0.0
        if joint_type == JointType.REVOLUTE:
            th = theta + offset
            ct = torch.cos(th)
            st = torch.sin(th)

        elif joint_type == JointType.PRISMATIC:
            d = d + offset + theta
            ct = np.cos(th)
            st = np.sin(th)
        else:
            assert joint_type == JointType.REVOLUTE and joint_type == JointType.PRISMATIC
            return

        trans = torch.tensor([[1.0,0,0,0], [0.0,1,0,0],[0,0,1,0],[0,0,0,1]], requires_grad=True).to(self.device)
        if convention == Convention.STANDARD:
            trans[0,0] = ct
            trans[0,1] = -st * ca
            trans[0,2] = st * sa
            trans[0,3] = a * ct
            trans[1,0] = st
            trans[1,1] = ct * ca
            trans[1,2] = -ct * sa
            trans[1,3] = a * st
            trans[2,0] = 0.0
            trans[2,1] = sa
            trans[2,2] = ca
            trans[2,3] = d

        elif convention == Convention.MODIFIED:
            trans[0,0] = ct
            trans[0,1] = -st
            trans[0,2] = 0
            trans[0,3] = a
            trans[1,0] = st * ca
            trans[1,1] = ct * ca
            trans[1,2] = -sa
            trans[1,3] = -d * sa
            trans[2,0] = st * sa
            trans[2,1] = ct * sa
            trans[2,2] = ca
            trans[2,3] = d * ca

        else:
            raise 'ERROR, DH CONVENTION NOT UNDERSTOOD'
        return trans

    
