# 这个是ctrgcn_stage1.py
import math
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from numpy import random
from graph.ntu_rgb_d import AdjMatrixGraph
from itertools import permutations
from apex import amp, optimizers
#from graph.ntu_rgb_d import AdjMatrixGraph, AdjMatrixGraph4lh, AdjMatrixGraph4rh, AdjMatrixGraph4ll, AdjMatrixGraph4rl, AdjMatrixGraph4torso


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # temporal convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # additional max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_tcn1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(unit_tcn1, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(0, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


# basic action feature learning and revised cross-view action feature learning
class cv_TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(cv_TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.gcn2 = unit_gcn(out_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=False)
        self.tcn2 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=1, dilations=dilations, residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, case, vlist): # or x, case, vlist, xr
        # case=0 means basic, case=1 means cross-view, vlist means view list！！！！！！！！！！！！
        if case == 0:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        else:
            # '''
            # CV-FCL (CV-STGCN) or GL-CVFD (only CV-GCN), the first V-FCL or C-CVFD training stage     
            temp = x
            # gcn1
            #print('0x.shape: ', x.shape) # ([20, 3, 100, 25])
            x = self.gcn1(x)
            #print('000x.shape: ', x.shape) # ([20, 64, 100, 25])
            # set batch size / gpus
            z = torch.zeros(16,2,x.size(1),x.size(2),x.size(3)).cuda().half()
            ##print('111x.shape: ', x.shape) # 
            # 本来CMC里面的111x是([32,64,100,25])，
            x = x.view(16,2,x.size(1),x.size(2),x.size(3)) # 
            ##print('2222x.shape: ', x.shape)
            ############300开始##########
            #z = torch.zeros(x.size(0),2,x.size(1),x.size(2),x.size(3)).cuda().half()
            
            ######################
            # cross-view spatial information interaction (25 joints, 50% probability)
            for i,view in enumerate(vlist):
                for j in range(25):
                    if random.choice([0,1]) == 0:
                        z[i,:,:,:,j] = x[i,:,:,:,j]
                    else:
                        z[i,:,:,:,j] = x[view,:,:,:,j]
            z = z.contiguous().view(16*2,x.size(2),x.size(3),x.size(4))
            # gcn2, tcn1
            z = self.tcn1(self.gcn2(z))
            # set batch size as 16
            m = torch.zeros(16,2,z.size(1),z.size(2),z.size(3)).cuda().half()
            z = z.view(16,2,z.size(1),z.size(2),z.size(3))
            # cross-view temporal information interaction (T frames, 50% probability)
            for i,view in enumerate(vlist):
                for j in range(z.size(3)):
                    if random.choice([0,1]) == 0:
                        m[i,:,:,j,:] = z[i,:,:,j,:]
                    else:
                        m[i,:,:,j,:] = z[view,:,:,j,:]
            m = m.contiguous().view(16*2,z.size(2),z.size(3),z.size(4))
            # tcn2
            y = self.relu(self.tcn2(m)+self.residual(temp))          
            #y = self.relu(self.tcn1(self.gcn2(z)) + self.residual(temp))
            '''
            # CV-FCL (CV-STGCN) or GL-CVFD (only CV-GCN), the second S-FCL or S-CVFD training stage   第二阶段！！！
            # only difference with above is the need of assistant views, others stay the same
            temp1 = x
            temp2 = xr
            # normal GCNs
            x = self.gcn1(x)
            xr = self.gcn1(xr)
            # batch size / gpus = 16
            z1 = torch.zeros(16,2,x.size(1),x.size(2),x.size(3)).cuda().half()
            z2 = torch.zeros(16,2,xr.size(1),xr.size(2),xr.size(3)).cuda().half()
            x = x.view(16,2,x.size(1),x.size(2),x.size(3))
            xr = xr.view(16,2,xr.size(1),xr.size(2),xr.size(3))
            # cross-view spatial information interaction (50% probability)
            for i in range(16):
                for j in range(25): # 25 joints, x is the main view, xr is the assistant view
                    if random.choice([0,1]) == 0:
                        z1[i,:,:,:,j] = x[i,:,:,:,j]
                    else:
                        z1[i,:,:,:,j] = xr[i,:,:,:,j]
            for i in range(16):
                for j in range(25): # 25 joints, xr is the main view, x is the assistant view
                    if random.choice([0,1]) == 0:
                        z2[i,:,:,:,j] = xr[i,:,:,:,j]
                    else:
                        z2[i,:,:,:,j] = x[i,:,:,:,j]

            z1 = z1.contiguous().view(16*2,x.size(2),x.size(3),x.size(4))
            z2 = z2.contiguous().view(16*2,xr.size(2),xr.size(3),xr.size(4))
            # CV-GCNs with normal TCNs
            z1 = self.tcn1(self.gcn2(z1))
            z2 = self.tcn1(self.gcn2(z2))
            m1 = torch.zeros(16,2,z1.size(1),z1.size(2),z1.size(3)).cuda().half()
            m2 = torch.zeros(16,2,z2.size(1),z2.size(2),z2.size(3)).cuda().half()
            z1 = z1.view(16,2,z1.size(1),z1.size(2),z1.size(3))
            z2 = z2.view(16,2,z2.size(1),z2.size(2),z2.size(3))
            # cross-view temporal information interaction (50% probability)
            for i in range(16):
                for j in range(z1.size(3)): # T frames, z1 is the main view, z2 is the assistant view
                    if random.choice([0,1]) == 0:
                        m1[i,:,:,j,:] = z1[i,:,:,j,:]
                    else:
                        m1[i,:,:,j,:] = z2[i,:,:,j,:]
            for i in range(16):
                for j in range(z2.size(3)): # T frames, z2 is the main view, z1 is the assistant view
                    if random.choice([0,1]) == 0:
                        m2[i,:,:,j,:] = z2[i,:,:,j,:]
                    else:
                        m2[i,:,:,j,:] = z1[i,:,:,j,:]

            m1 = m1.contiguous().view(16*2,z1.size(2),z1.size(3),z1.size(4))
            m2 = m2.contiguous().view(16*2,z2.size(2),z2.size(3),z2.size(4))
            # CV-TCNs
            y1 = self.relu(self.tcn2(m1)+self.residual(temp1))
            y2 = self.relu(self.tcn2(m2)+self.residual(temp2))
            #y1 = self.relu(self.tcn1(self.gcn2(z1)) + self.residual(temp1))
            #y2 = self.relu(self.tcn1(self.gcn2(z2)) + self.residual(temp2))
            '''

        return y
        # return y1,y2

        
class TCN_GCN_unit1(nn.Module):
    def __init__(self, in_channels, out_channels, A, residual=True, adaptive=True, kernel_size=1, stride=1):
        super(TCN_GCN_unit1, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = unit_tcn1(out_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        x = x * x_att
        return x


class Model(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True):
        super(Model, self).__init__()

        Graph = AdjMatrixGraph()
        A = Graph.A_binary

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = cv_TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive) # 
        self.att1 = ST_Joint_Att(base_channel, 4, True) # ##
        self.l2 = cv_TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = cv_TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive) # ##
        self.l4 = cv_TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive) # ##
        self.l5 = cv_TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.att2 = ST_Joint_Att(base_channel*2, 4, True) # ##
        self.l6 = cv_TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = cv_TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive) # ##
        self.l8 = cv_TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.att3 = ST_Joint_Att(base_channel*4, 4, True) # ##
        self.l9 = cv_TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive) # ##
        self.l10 = cv_TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        #self.cvgcn1 = unit_gcn(in_channels, base_channel, A, adaptive=adaptive) # 
        #self.cvgcn2 = unit_gcn(base_channel, base_channel, A, adaptive=adaptive)
        #self.cvgcn3 = unit_gcn(base_channel, base_channel, A, adaptive=adaptive)
        #self.cvgcn4 = unit_gcn(base_channel, base_channel, A, adaptive=adaptive)
        #self.cvgcn5 = unit_gcn(base_channel, base_channel*2, A, adaptive=adaptive)
        #self.cvgcn6 = unit_gcn(base_channel*2, base_channel*2, A, adaptive=adaptive)
        #self.cvgcn7 = unit_gcn(base_channel*2, base_channel*2, A, adaptive=adaptive)
        #self.cvgcn8 = unit_gcn(base_channel*2, base_channel*4, A, adaptive=adaptive)
        #self.cvgcn9 = unit_gcn(base_channel*4, base_channel*4, A, adaptive=adaptive)

        # original classifier
        #self.fc = nn.Linear(base_channel*4, num_class)
        # transform feature-dim from 256 to 128 for view-semantic Fisher contrastive learning
        self.fc = nn.Linear(base_channel*4, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU(True)
        # transform feature-dim from 128 to 40/8, 60/3, 120/3 for semantic/view discrimination
        self.prototypes = nn.Linear(128, 40, bias=False)
        self.prototypev = nn.Linear(128, 8, bias=False)
        
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / 128))
        nn.init.normal_(self.prototypev.weight, 0, math.sqrt(2. / 8))
        nn.init.normal_(self.prototypes.weight, 0, math.sqrt(2. / 40))
        
        bn_init(self.data_bn, 1)
        bn_init(self.bn, 1)
        bn_init(self.bn1, 1)

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x): # x (V-FCL) or x, xr (S-FCL)      
        N, C, T, V, M = x.size()
        #print('509  x.shape: ', x.shape) # ([10, 3, 100, 25, 2])

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # # assistant view
        # xr = xr.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # xr = self.data_bn(xr)
        # xr = xr.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # corrsponding to batch size / gpus in cv_TCN_GCN_unit
        vlist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        random.shuffle(vlist)
        #print(vlist)
        xcv = x
        # xcv1 = xr  # 
        # 

        #print('--528-- x.shape: ', x.shape) # ([20, 3, 100, 25])
        x = self.l1(x,0,vlist) # only in the first view-term training stage # stage1 debug
        #print('----528---- x.shape: ', x.shape) # ([20, 64, 100, 25])
        xcv = self.l1(xcv,1,vlist) # or xcv,xcv1 = self.l1(xcv,1,vlist,xcv1) for the second semantic-term training stage
        # 
        # xcv,xcv1 = self.l1(xcv,1,vlist,xcv1) # 
        x = self.att1(x) # 
        x = self.l2(x,0,vlist) # 
        xcv = self.l2(xcv,1,vlist)
        # xcv,xcv1 = self.l2(xcv,1,vlist,xcv1) # ##
        x = self.l3(x,0,vlist) # 
        xcv = self.l3(xcv,1,vlist)
        x = self.l4(x,0,vlist) # 
        xcv = self.l4(xcv,1,vlist)
        x = self.l5(x,0,vlist) # 
        xcv = self.l5(xcv,1,vlist)
        # xcv,xcv1 = self.l5(xcv,1,vlist,xcv1) # ##
        x = self.att2(x) # 
        x = self.l6(x,0,vlist) #
        xcv = self.l6(xcv,1,vlist)
        # xcv,xcv1 = self.l6(xcv,1,vlist,xcv1) # ##
        x = self.l7(x,0,vlist) # 
        xcv = self.l7(xcv,1,vlist)
        x = self.l8(x,0,vlist) # 
        xcv = self.l8(xcv,1,vlist)
        # xcv,xcv1 = self.l8(xcv,1,vlist,xcv1) # ##
        x = self.att3(x) # 
        x = self.l9(x,0,vlist) # 
        xcv = self.l9(xcv,1,vlist)
        x = self.l10(x,0,vlist) # 
        xcv = self.l10(xcv,1,vlist)
        # xcv,xcv1 = self.l10(xcv,1,vlist,xcv1) # 
        # original version, N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        #temp = x

        # the first view-term training stage     
        c_new1 = xcv.size(1)
        xcv = xcv.view(N, M, c_new1, -1)
        xcv = xcv.mean(3).mean(1)
        xcv = self.drop_out(xcv)
        tempcv = xcv # view-common feature
        tempv = x-xcv # view-specific feature  

        # # the second semantic-term training stage, assistant view      
        # c_new2 = xcv1.size(1)
        # xcv1 = xcv1.view(N, M, c_new2, -1)
        # xcv1 = xcv1.mean(3).mean(1)
        # xcv1 = self.drop_out(xcv1)
        # tempcv1 = xcv1 # assistant-view view-common feature

        # 
        outv = self.relu1(self.bn1(tempv)) # 
        outv = self.fc(outv) # same-action different-view view-specific feature
        outp = self.relu1(self.bn1(tempcv))
        outp = self.fc(outp) # same-action different-view view-common feature or different-action view-common feature
        fv = self.relu(self.bn(outv))
        fv = self.prototypev(fv) # view discrimination vector based on outv

        # 
        # fp = self.relu(self.bn(outp)) # 
        # fp = self.prototypes(fp) # semantic discrimination vector based on outp
        # outp1 = self.relu1(self.bn1(tempcv1)) #
        # outp1 = self.fc(outp1) # different-action assistant-view view-common feature
        # fp1 = self.relu(self.bn(outp1)) # 
        # fp1 = self.prototypes(fp1) # semantic discrimination vector based on outp1

        return outv,outp,fv
        # return outp, outp1, fp, fp1


# Model1 is for skeleton reconstruction (pretext task learning)
class Model1(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True):
        super(Model1, self).__init__()

        Graph = AdjMatrixGraph()
        A = Graph.A_binary

        self.num_class = num_class
        self.num_point = num_point
        #self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        # cv_TCN_GCN_unit uses case=0
        self.l1 = TCN_GCN_unit1(base_channel, in_channels, A, residual=False, adaptive=adaptive)
        self.l2 = cv_TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = cv_TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = cv_TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit1(base_channel*2, base_channel, A, residual=False, adaptive=adaptive, kernel_size=2, stride=2)
        self.l6 = cv_TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = cv_TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit1(base_channel*4, base_channel*2, A, residual=False, adaptive=adaptive, kernel_size=2, stride=2)
        self.l9 = cv_TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = cv_TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.att1 = ST_Joint_Att(base_channel*4, 4, True)
        self.att2 = ST_Joint_Att(base_channel*2, 4, True)
        self.att3 = ST_Joint_Att(base_channel, 4, True)

    def forward(self, xp):

        xp = self.l10(xp)
        xp = self.att1(xp)
        xp = self.l9(xp)
        xp = self.l8(xp)
        xp = self.att2(xp)
        xp = self.l7(xp)
        xp = self.l6(xp)
        xp = self.l5(xp)
        xp = self.att3(xp)
        xp = self.l4(xp)
        xp = self.l3(xp)
        xp = self.l2(xp)
        xp = self.l1(xp)

        return xp


# Model2 is for view transformation network
class Model2(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True):
        super(Model2, self).__init__()

        Graph = AdjMatrixGraph()
        A = Graph.A_binary

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        # cv_TCN_GCN_unit uses case=0
        self.l1 = cv_TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = cv_TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = cv_TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 6)
        
        nn.init.normal_(self.fc1.weight, 0, math.sqrt(2. / 32))
        nn.init.normal_(self.fc2.weight, 0, math.sqrt(2. / 6))
        
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        
        temp = x
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        trans = x
        
        temp = temp.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C*V*M)
        x = _transform(temp,trans)

        return x,trans
        
class CTRGCN(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True):
        super(CTRGCN, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(150)

        self.encoderp_s = Model(num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True)
        #self.decoder = Model1(num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True)
        #self.transp_s = Model2(num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True)

        bn_init(self.data_bn, 1)
        #self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

    def forward(self, x, viewt):

        xr = _transform_rotation(x) # assistant view
        x = _transform_shape(x)
        N, C, T, V, M = x.size()
        #x1 = torch.split(x, [int(N/2),int(N/2)], dim=0)[0]
        #x2 = torch.split(x, [int(N/2),int(N/2)], dim=0)[1]
        xjmo = torch.zeros(N, C, T, V, M).cuda().half() # joint motion
        xjmor = torch.zeros(N, C, T, V, M).cuda().half() # assistant-view joint motion
        xnew = torch.zeros(N, C, T, V, M).cuda().half() # new representation stream
        for i in range(T-1):
            xjmo[:,:,i,:,:] = x[:,:,i+1,:,:] - x[:,:,i,:,:]
            xjmor[:,:,i,:,:] = xr[:,:,i+1,:,:] - xr[:,:,i,:,:]
        xnew[:,0,:,:,:] = torch.sqrt(torch.pow(xjmo[:,0,:,:,:],2)+torch.pow(xjmo[:,1,:,:,:],2)+torch.pow(xjmo[:,2,:,:,:],2))
        xnew[:,1,:,:,:] = torch.acos(xjmo[:,2,:,:,:]/(torch.sqrt(torch.pow(xjmo[:,0,:,:,:],2)+torch.pow(xjmo[:,1,:,:,:],2)+torch.pow(xjmo[:,2,:,:,:],2))))
        xnew[:,2,:,:,:] = torch.atan(xjmo[:,1,:,:,:]/xjmo[:,0,:,:,:])

        '''
        # different spatio-temporal shuffle strategies for pretext tasks
        choice = random.randint(1,5)
        if choice == 1:
            if random.randint(0,1)==0:
                xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4,5,6,7,8]).cuda()), torch.zeros(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([12,13,14,15,16,17,18,19,20,21,22]).cuda()), torch.zeros(N,C,T,2,M).cuda()), dim=3)
            else:
                xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4,5,6,7,8]).cuda()), torch.randn(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([12,13,14,15,16,17,18,19,20,21,22]).cuda()), torch.randn(N,C,T,2,M).cuda()), dim=3)
        
        if choice == 2:
            if random.randint(0,1)==0:
                xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4]).cuda()), torch.zeros(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([8,9,10,11,12,13,14,15,16,17,18,19,20]).cuda()), torch.zeros(N,C,T,2,M).cuda(), torch.index_select(x, 3, torch.tensor([23,24]).cuda())), dim=3)
            else:
                xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4]).cuda()), torch.randn(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([8,9,10,11,12,13,14,15,16,17,18,19,20]).cuda()), torch.randn(N,C,T,2,M).cuda(), torch.index_select(x, 3, torch.tensor([23,24]).cuda())), dim=3)
        
        if choice == 3:
            if random.randint(0,1)==0:
                xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).cuda()), torch.zeros(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([20,21,22,23,24]).cuda())), dim=3)
            else:
                xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).cuda()), torch.randn(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([20,21,22,23,24]).cuda())), dim=3)
        
        if choice == 4:
            if random.randint(0,1)==0:
                xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12]).cuda()), torch.zeros(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([16,17,18,19,20,21,22,23,24]).cuda())), dim=3)
            else:
                xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12]).cuda()), torch.randn(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([16,17,18,19,20,21,22,23,24]).cuda())), dim=3)
        
        if choice == 5:
            if random.randint(0,1)==0:
                xj = torch.cat((torch.zeros(N,C,T,4,M).cuda(), torch.index_select(x, 3, torch.tensor([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).cuda()), torch.zeros(N,C,T,1,M).cuda(), torch.index_select(x, 3, torch.tensor([21,22,23,24]).cuda())), dim=3)
            else:
                xj = torch.cat((torch.randn(N,C,T,4,M).cuda(), torch.index_select(x, 3, torch.tensor([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).cuda()), torch.randn(N,C,T,1,M).cuda(), torch.index_select(x, 3, torch.tensor([21,22,23,24]).cuda())), dim=3)
        
        xj = torch.cat((xj[:,:,:50,:,:], torch.randn(N,C,5,V,M).cuda(), xj[:,:,55:150,:,:], torch.randn(N,C,5,V,M).cuda(), torch.zeros(N,C,345,V,M).cuda()), dim=2)
        '''

        '''
        # skeleton reconstruction (pretext task learning)
        out1vr = outvr[0:int(N/2),:]
        out2vr = outvr[int(N/2):N,:]
        out1pr = outpr[0:int(N/2),:]
        out2pr = outpr[int(N/2):N,:]
        out1r = torch.add(out1vr,out2pr)
        out2r = torch.add(out2vr,out1pr)
        out1 = self.decoder(torch.cat((out1r,out1r), dim=0))
        out2 = self.decoder(torch.cat((out2r,out2r), dim=0))
        out = torch.cat((out1, out2), dim=0)
        '''

        '''
        # Barlow Twins (contrastive learning)
        labellist = list()
        for i in range(int(N/2)):
            llist2 = list()
            llist1 = list()
            for j in range(int(N/2)):
                
                if t[i] == t[int(N/2)+j]:
                #simp = torch.cosine_similarity(out1_p[:,i],out2_p[:,j],dim=0).cuda()
                #simv = torch.cosine_similarity(out1_v[:,i],out2_v[:,j],dim=0).cuda()
                #if i == j:
                #count = count + 1
                    #diffp = diffp + simp.add_(-1).pow_(2)
                    #diffv = diffv + simv.pow_(2)
                    llist1.append(1)
                
                    if viewt[i] == viewt[int(N/2)+j]:
                    #diffv = diffv + simv.add_(-1).pow_(2)
                    #diffp = diffp + simp.pow_(2)
                        llist2.append(1)
                    else:
                        llist2.append(0)
            
            labellist.append(llist1+llist2)
        
        label = torch.tensor(labellist).cuda()
        cp = out1_p.mm(out2_p.t())/0.2
        cv = out1_v.mm(out2_v.t())/0.2
        diff = torch.cat((cp,cv),dim=1)
        '''

        # 
        outv, outp, fv = self.encoderp_s(xjmo) # 
        # outp, outp1, fp, fp1 = self.encoderp_s(xjmo, xjmor) # xjmo (V-FCL) or xjmo, xjmor (S-FCL) # 

        return outv, outp, fv
        # return outp, outp1, fp, fp1
        
#
class MyCTRGCN(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True):
        super(MyCTRGCN, self).__init__()
        self.net = CTRGCN(num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True)

        self.net = torch.nn.DataParallel(self.net)
        #self.net = torch.nn.parallel.DistributedDataParallel(self.net) # model distributed training setting
               
    def forward(self, x, viewt):
        # 
        return self.net(x, viewt)
        
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

# manually set view transformation parameters to generate assistant views
def _transform_rotation(inputs1):
    view = list()
    view1 = [math.pi/8,2*math.pi/8,3*math.pi/8,4*math.pi/8,5*math.pi/8,6*math.pi/8,7*math.pi/8,8*math.pi/8,9*math.pi/8,10*math.pi/8,11*math.pi/8,12*math.pi/8,13*math.pi/8,14*math.pi/8,15*math.pi/8]
    a1, a2 = inputs1.split([75, 75], dim=2)
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25 = a1.chunk(25, dim=2)
    inputs1 = torch.stack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25], dim=3)
    inputs1 = inputs1.permute(0, 2, 1, 3)

    x = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            #x[i][j] = random.randrange(0,math.pi/4,math.pi/36)
            #x[i][j] = random.choice([0,math.pi/36,math.pi/18,3*math.pi/36,math.pi/9,5*math.pi/36,6*math.pi/36,7*math.pi/36,8*math.pi/36,math.pi/4])
            x[i][j] = 0 # no rotation
    y = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            #y[i][j] = random.randrange(0,2*math.pi,math.pi/8)
            viewchoice = random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
            view.append(viewchoice)
            y[i][j] = view1[viewchoice] # random rotation based on viewchoice
    z = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            #z[i][j] = random.randrange(0,math.pi/4,math.pi/36)
            #z[i][j] = random.choice([0,math.pi/36,math.pi/18,3*math.pi/36,math.pi/9,5*math.pi/36,6*math.pi/36,7*math.pi/36,8*math.pi/36,math.pi/4])
            z[i][j] = 0 # no rotation
    rot_x = torch.tensor(x)
    rot_y = torch.tensor(y)
    rot_z = torch.tensor(z)
    rot = torch.cat((rot_x,rot_y,rot_z),dim=1).float().cuda()
    t = np.empty([inputs1.shape[0],3], dtype = float) 
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            t[i][j] = random.uniform(0,0)
    trans = torch.tensor(t).half().cuda()
    trans, rot = _trans_rot(trans, rot)
    inputs1 = inputs1.contiguous().view(-1, inputs1.size()[1], inputs1.size()[2] * inputs1.size()[3])
    inputs1 = torch.cat((inputs1, Variable(inputs1.data.new(inputs1.size()[0], 1, inputs1.size()[2]).fill_(1))), dim=1).cuda()
    inputs1 = torch.matmul(trans.half(), inputs1)
    inputs1 = torch.matmul(rot.half(), inputs1)
    inputs1 = inputs1.contiguous().view(-1, 3, 3014, 25) # max frame 3014
    inputs1 = torch.split(inputs1, [300,2714], dim=2)[0] # split 3014 frames
    #inputs1 = inputs1.contiguous().view(-1, 3, 300, 25)
    inputs1 = torch.index_select(inputs1, 2, torch.arange(1,200,2).cuda()) # frame downsampling

    '''
    x = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            #x[i][j] = random.uniform(0,math.pi/4)
            x[i][j] = 0
    y = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            #y[i][j] = random.uniform(0,2*math.pi)
            y[i][j] = random.choice([0,math.pi/8,2*math.pi/8,3*math.pi/8,4*math.pi/8,5*math.pi/8,6*math.pi/8,7*math.pi/8,8*math.pi/8,9*math.pi/8,10*math.pi/8,11*math.pi/8,12*math.pi/8,13*math.pi/8,14*math.pi/8,15*math.pi/8])
    z = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            #z[i][j] = random.uniform(0,math.pi/4)
            z[i][j] = 0
            
    rot_x = torch.tensor(x)
    rot_y = torch.tensor(y)
    rot_z = torch.tensor(z)
    rot = torch.cat((rot_x, rot_y, rot_z), dim=1).float().cuda()
    t = np.empty([inputs1.shape[0],3], dtype = float) 
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            t[i][j] = random.uniform(0,0)
    trans = torch.tensor(t).float().cuda()
    trans, rot = _trans_rot(trans, rot)
    '''

    b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55 = a2.chunk(25, dim=2)
    inputs2 = torch.stack([b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55], dim=3)
    inputs2 = inputs2.permute(0, 2, 1, 3)
    inputs2 = inputs2.contiguous().view(-1, inputs2.size()[1], inputs2.size()[2] * inputs2.size()[3])
    inputs2 = torch.cat((inputs2, Variable(inputs2.data.new(inputs2.size()[0], 1, inputs2.size()[2]).fill_(1))), dim=1).cuda()
    inputs2 = torch.matmul(trans.half(), inputs2)
    inputs2 = torch.matmul(rot.half(), inputs2)
    inputs2 = inputs2.contiguous().view(-1, 3, 3014, 25)
    inputs2 = torch.split(inputs2, [300,2714], dim=2)[0]
    #inputs2 = inputs2.contiguous().view(-1, 3, 300, 25)
    inputs2 = torch.index_select(inputs2, 2, torch.arange(1,200,2).cuda())

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

    '''
    x = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            #x[i][j] = random.uniform(0,math.pi/4)
            x[i][j] = 0
    y = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            #y[i][j] = random.uniform(0,2*math.pi)
            y[i][j] = random.choice([0,math.pi/8,2*math.pi/8,3*math.pi/8,4*math.pi/8,5*math.pi/8,6*math.pi/8,7*math.pi/8,8*math.pi/8,9*math.pi/8,10*math.pi/8,11*math.pi/8,12*math.pi/8,13*math.pi/8,14*math.pi/8,15*math.pi/8])
    z = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            #z[i][j] = random.uniform(0,math.pi/4)
            z[i][j] = 0
            
    rot_x = torch.tensor(x)
    rot_y = torch.tensor(y)
    rot_z = torch.tensor(z)
    rot = torch.cat((rot_x, rot_y, rot_z), dim=1).float().cuda()
    t = np.empty([inputs1.shape[0],3], dtype = float) 
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            t[i][j] = random.uniform(0,0)
    trans = torch.tensor(t).float().cuda()
    trans, rot = _trans_rot(trans, rot)
    '''

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
    inputs1 = torch.split(inputs1, [300,2714], dim=2)[0]
    #inputs1 = inputs1.contiguous().view(-1, 3, 300, 25)
    inputs1 = torch.index_select(inputs1, 2, torch.arange(1,200,2).cuda())
    b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55 = a2.chunk(25, dim=2)
    inputs2 = torch.stack([b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55], dim=3)
    inputs2 = inputs2.permute(0, 2, 1, 3)
    inputs2 = inputs2.contiguous().view(-1, 3, 3014, 25)
    inputs2 = torch.split(inputs2, [300,2714], dim=2)[0]
    #inputs2 = inputs2.contiguous().view(-1, 3, 300, 25)
    inputs2 = torch.index_select(inputs2, 2, torch.arange(1,200,2).cuda())
    inputs = torch.stack([inputs1, inputs2], dim=4)

    return inputs