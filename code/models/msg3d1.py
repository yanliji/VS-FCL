import sys
sys.path.insert(0, '')
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import import_class, count_params
from models.ms_gcn import MultiScale_GraphConv as MS_GCN
from models.ms_tcn import MultiScale_TemporalConv as MS_TCN
from models.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from models.mlp import MLP
from models.activation import activation_factory
from torch.autograd import Variable
from numpy import random
import numpy as np
import math
from graph.ntu_rgb_d import AdjMatrixGraph
from models.ms_tcn import TemporalConv
from models.ms_tcn import TemporalDeConv
from itertools import permutations
import random


class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        x = self.gcn3d(x)

        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        return x


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3,5],
                 window_stride=1,
                 window_dilations=[1,1]):

        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)

        return out_sum
 
   
class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model, self).__init__()

        Graph = AdjMatrixGraph()
        A_binary = Graph.A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        c1 = 96
        c2 = c1 * 2
        c3 = c2 * 2

        self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

    def forward(self, x):     
        #xv = _transform_rotation(x)
        N, C, T, V, M = x.size()
     
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)
        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)
        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)

        out1 = x     
        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)
        out = out.mean(1) 

        '''
        xv = xv.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        xv = self.data_bn(xv)
        xv = xv.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        
        xv = F.relu(self.sgcn1(xv) + self.gcn3d1(xv), inplace=True)
        xv = self.tcn1(xv)
        xv = F.relu(self.sgcn2(xv) + self.gcn3d2(xv), inplace=True)
        xv = self.tcn2(xv)
        xv = F.relu(self.sgcn3(xv) + self.gcn3d3(xv), inplace=True)
        xv = self.tcn3(xv)
             
        outv = xv
        out_channels = outv.size(1)
        outv = outv.view(N, M, out_channels, -1)
        outv = outv.mean(3)   
        outv = outv.mean(1)   
        outv = self.fc(outv)
        '''
        
        return out

     
class Model1(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model1, self).__init__()

        Graph = AdjMatrixGraph()
        A_binary = Graph.A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        c1 = 96
        c2 = c1 * 2
        c3 = c2 * 2

        self.gcn3d1r = MultiWindow_MS_G3D(c1, 3, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1r = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, 3, A_binary, disentangled_agg=True),
            TemporalConv(3, 3),
            TemporalConv(3, 3))
        self.sgcn1r[-1].act = nn.Identity()
        self.tcn1r = MS_TCN(c1, c1)

        self.gcn3d2r = MultiWindow_MS_G3D(c2, c1, A_binary, num_g3d_scales, window_stride=1)
        self.tcn22r = TemporalDeConv(c1,c1)
        self.sgcn2r = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            TemporalDeConv(c2, c1),
            MS_TCN(c1, c1))
        self.sgcn2r[-1].act = nn.Identity()
        self.tcn2r = MS_TCN(c2, c2)

        self.gcn3d3r = MultiWindow_MS_G3D(c3, c2, A_binary, num_g3d_scales, window_stride=1)
        self.tcn33r = TemporalDeConv(c2,c2)
        self.sgcn3r = nn.Sequential(
            MS_GCN(num_gcn_scales, c3, c3, A_binary, disentangled_agg=True),
            TemporalDeConv(c3, c2),
            MS_TCN(c2, c2))
        self.sgcn3r[-1].act = nn.Identity()
        self.tcn3r = MS_TCN(c3, c3)

        #self.gcnr = MS_GCN(num_gcn_scales, c3, c2, A_binary, disentangled_agg=True)
        #self.gcnr1 = MS_GCN(num_gcn_scales, c2, c1, A_binary, disentangled_agg=True)
        #self.gcnr2 = MS_GCN(num_gcn_scales, c1, 3, A_binary, disentangled_agg=True)
        #self.tcnr = TemporalDeConv(3, 3)
        #self.tcnr1 = TemporalDeConv(3, 3)

    def forward(self, xp):
        '''
        xp = self.gcnr(xp)
        xp = self.gcnr1(xp)
        xp = self.gcnr2(xp)
        xp = self.tcnr(xp)
        xp = self.tcnr1(xp)
        '''

        xp = self.tcn3r(xp)
        #x = self.sgcn3r(xp)
        #y = self.gcn3d3r(xp)

        xp = F.relu(self.sgcn3r(xp) + self.tcn33r(self.gcn3d3r(xp)), inplace=True)
        xp = self.tcn2r(xp)
        xp = F.relu(self.sgcn2r(xp) + self.tcn22r(self.gcn3d2r(xp)), inplace=True)
        xp = self.tcn1r(xp)
        xp = F.relu(self.sgcn1r(xp) + self.gcn3d1r(xp), inplace=True)

        return xp
        

class Model2(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model2, self).__init__()

        Graph = AdjMatrixGraph()
        A_binary = Graph.A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        c1 = 96
        c2 = c1 * 2

        self.gcn1 = MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True)
        self.gcn2 = MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True)
        self.tcn1 = MS_TCN(c1, c1)
        self.tcn2 = MS_TCN(c1, c2, stride=2)
        self.fc1 = nn.Linear(c2, c1)
        self.fc2 = nn.Linear(c1, 32)
        self.fc3 = nn.Linear(32, 6)

    def forward(self, x):     
        temp = x
        #x = _transform_shape(x)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        
        x = self.gcn1(x)
        x = self.tcn1(x)
        x = F.relu(x)
        x = self.gcn2(x)
        x = self.tcn2(x)
        x = F.relu(x)
        out_channels = x.size(1)
        x = x.view(N, M, out_channels, -1)
        x = x.mean(3)   
        x = x.mean(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        trans = x
        
        temp = temp.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C*V*M)
        x = _transform(temp,trans)
        
        return x,trans
        

class Msg3d(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Msg3d, self).__init__()
        self.encoderp_s = Model(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3)
        '''
        self.encoderv_s = Model(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3)
        '''
        self.transp_s = Model2(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3)
        '''
        self.decoder = Model1(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3)
        '''
        #self.data_bn = nn.BatchNorm1d(150)

    def forward(self, x):
        '''
        x = _transform_shape(x)
        N, C, T, V, M = x.size()
        
        x1 = torch.split(x, [int(N/2),int(N/2)], dim=0)[0]
        x2 = torch.split(x, [int(N/2),int(N/2)], dim=0)[1]
        out1_gt = x1
        out2_gt = x2
        out1_gt = out1_gt.permute(0, 4, 3, 1, 2).contiguous().view(int(N/2), M * V * C, T)
        out1_gt = self.data_bn(out1_gt)
        out1_gt = out1_gt.view(int(N/2) * M, V, C, T).permute(0,2,3,1).contiguous()
        out2_gt = out2_gt.permute(0, 4, 3, 1, 2).contiguous().view(int(N/2), M * V * C, T)
        out2_gt = self.data_bn(out2_gt)
        out2_gt = out2_gt.view(int(N/2) * M, V, C, T).permute(0,2,3,1).contiguous()
        
        out1vr,out1v = self.encoderv_s(x1)
        x1,trans = self.transp_s(x1)
        out1pr,out1p = self.encoderp_s(x1)
        out1vr = torch.abs(out1vr-out1pr)
        
        out2vr,out2v = self.encoderv_s(x2)
        x2,trans = self.transp_s(x2)
        out2pr,out2p = self.encoderp_s(x2)
        out2vr = torch.abs(out2vr-out2pr)
        
        out1r = out1vr+out2pr
        out2r = out2vr+out1pr
        out1 = self.decoder(out1r)
        out2 = self.decoder(out2r)
        '''
        
        x = _transform_shape(x)
        #x1 = self.encoderv_s(x)
        x2,trans = self.transp_s(x)
        x2 = self.encoderp_s(x2)
        #x1 = torch.abs(x1-x2)
        return x2

        #out_gt = torch.cat((out1_gt,out2_gt),dim=0)
        #out = torch.cat((out1,out2),dim=0)
        #return out_gt, out

class MyMsg3d(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(MyMsg3d, self).__init__()
        self.net = Msg3d(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3)

        self.net = torch.nn.DataParallel(self.net)
               
    def forward(self, x):       
        return self.net(x)


def _trans_rot(trans, rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = Variable(rot.data.new(rot.size()[:1] + (1,)).zero_())
    ones = Variable(rot.data.new(rot.size()[:1] + (1,)).fill_(1))

    r1 = torch.stack((ones, zeros, zeros),dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2, rx3), dim = 1)

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)

    rt1 = torch.stack((ones, zeros, zeros, trans[:,0:1]), dim = -1)
    rt2 = torch.stack((zeros, ones, zeros, trans[:,1:2]), dim = -1)
    rt3 = torch.stack((zeros, zeros, ones, trans[:,2:3]), dim = -1)
    trans = torch.cat((rt1, rt2, rt3), dim = 1)

    return trans, rot


def _transform_rotation(inputs1):
    a1, a2 = inputs1.split([75, 75], dim=2)
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25 = a1.chunk(25, dim=2)
    inputs1 = torch.stack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25], dim=3)
    inputs1 = inputs1.permute(0, 2, 1, 3)

    x = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = random.uniform(0,math.pi/4)
    y = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i][j] = random.uniform(0,2*math.pi)
    z = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i][j] = random.uniform(0,math.pi/4)
    
    rot_x = torch.tensor(x)
    rot_y = torch.tensor(y)
    rot_z = torch.tensor(z)
    rot = torch.cat((rot_x,rot_y,rot_z),dim=1).float().cuda()
    t = np.empty([inputs1.shape[0],3], dtype = float) 
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            t[i][j] = random.uniform(0,0.1)
    trans = torch.tensor(t).float().cuda()
    trans, rot = _trans_rot(trans, rot)
    inputs1 = inputs1.contiguous().view(-1, inputs1.size()[1], inputs1.size()[2] * inputs1.size()[3])
    inputs1 = torch.cat((inputs1, Variable(inputs1.data.new(inputs1.size()[0], 1, inputs1.size()[2]).fill_(1))), dim=1).cuda()
    inputs1 = torch.matmul(trans, inputs1)
    inputs1 = torch.matmul(rot, inputs1)
    inputs1 = inputs1.contiguous().view(-1, 3, 300, 25)

    b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55 = a2.chunk(25, dim=2)
    inputs2 = torch.stack([b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55], dim=3)
    inputs2 = inputs2.permute(0, 2, 1, 3)
    
    x = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = random.uniform(0,math.pi/4)
    y = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i][j] = random.uniform(0,2*math.pi)
    z = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i][j] = random.uniform(0,math.pi/4)
            
    rot_x = torch.tensor(x)
    rot_y = torch.tensor(y)
    rot_z = torch.tensor(z)
    rot = torch.cat((rot_x, rot_y, rot_z), dim=1).float().cuda()
    t = np.empty([inputs1.shape[0],3], dtype = float) 
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            t[i][j] = random.uniform(0,0.1)
    trans = torch.tensor(t).float().cuda()
    trans, rot = _trans_rot(trans, rot)
    inputs2 = inputs2.contiguous().view(-1, inputs2.size()[1], inputs2.size()[2] * inputs2.size()[3])
    inputs2 = torch.cat((inputs2, Variable(inputs2.data.new(inputs2.size()[0], 1, inputs2.size()[2]).fill_(1))), dim=1).cuda()
    inputs2 = torch.matmul(trans, inputs2)
    inputs2 = torch.matmul(rot, inputs2)
    inputs2 = inputs2.contiguous().view(-1, 3, 300, 25)

    inputs = torch.stack([inputs1, inputs2], dim=4)

    return inputs

   
def _transform(inputs1,mat):
    
    rot = mat[:,0:3]
    trans = mat[:,3:6]
    
    a1, a2 = inputs1.split([75, 75], dim=2)
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25 = a1.chunk(25, dim=2)
    inputs1 = torch.stack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25], dim=3)
    inputs1 = inputs1.permute(0, 2, 1, 3)
    
    trans, rot = _trans_rot(trans, rot)
    inputs1 = inputs1.contiguous().view(-1, inputs1.size()[1], inputs1.size()[2] * inputs1.size()[3])
    inputs1 = torch.cat((inputs1, Variable(inputs1.data.new(inputs1.size()[0], 1, inputs1.size()[2]).fill_(1))), dim=1).cuda()
    inputs1 = torch.matmul(trans, inputs1)
    inputs1 = torch.matmul(rot, inputs1)
    inputs1 = inputs1.contiguous().view(-1, 3, 500, 25)
    #inputs1 = torch.split(inputs1, [500,2514], dim=2)[0]

    b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55 = a2.chunk(25, dim=2)
    inputs2 = torch.stack([b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55], dim=3)
    inputs2 = inputs2.permute(0, 2, 1, 3)
    inputs2 = inputs2.contiguous().view(-1, inputs2.size()[1], inputs2.size()[2] * inputs2.size()[3])
    inputs2 = torch.cat((inputs2, Variable(inputs2.data.new(inputs2.size()[0], 1, inputs2.size()[2]).fill_(1))), dim=1).cuda()
    inputs2 = torch.matmul(trans, inputs2)
    inputs2 = torch.matmul(rot, inputs2)
    inputs2 = inputs2.contiguous().view(-1, 3, 500, 25)
    #inputs2 = torch.split(inputs2, [500,2514], dim=2)[0]

    inputs = torch.stack([inputs1, inputs2], dim=4)

    return inputs


def _transform_shape(inputs1):
    a1, a2 = inputs1.split([75, 75], dim=2)
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25 = a1.chunk(25, dim=2)
    inputs1 = torch.stack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25], dim=3)
    inputs1 = inputs1.permute(0, 2, 1, 3)
    inputs1 = inputs1.contiguous().view(-1, 3, 3014, 25)
    inputs1 = torch.split(inputs1, [500,2514], dim=2)[0]
    b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55 = a2.chunk(25, dim=2)
    inputs2 = torch.stack([b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55], dim=3)
    inputs2 = inputs2.permute(0, 2, 1, 3)
    inputs2 = inputs2.contiguous().view(-1, 3, 3014, 25)
    inputs2 = torch.split(inputs2, [500,2514], dim=2)[0]
    inputs = torch.stack([inputs1, inputs2], dim=4)

    return inputs
    

if __name__ == "__main__":

    import sys
    sys.path.append('..')

    model = Model(
        num_class=40,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 8, 3, 50, 25, 2
    x = torch.randn(N,C,T,V,M)
    model.forward(x)

    print('Model total # params:', count_params(model))
