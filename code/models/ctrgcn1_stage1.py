#ctrgcn_stage1.py
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
            # CV-FCL (CV-STGCN) or GL-CVFD (only CV-GCN), the first V-FCL or C-CVFD training stage     第一阶段！！！
            temp = x # x.shape = ([20, 3, 100, 25])
            x = self.gcn1(x)
            # set batch size / gpus             
            z = torch.zeros(16,2,x.size(1),x.size(2),x.size(3)).cuda().half()
            x = x.view(16,2,x.size(1),x.size(2),x.size(3))
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

        return y

        
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
        xcv = x
        # xcv1 = xr  # 

        x = self.l1(x,0,vlist)
        xcv = self.l1(xcv,1,vlist)
        x = self.att1(x)
        x = self.l2(x,0,vlist)
        xcv = self.l2(xcv,1,vlist)
        x = self.l3(x,0,vlist)
        xcv = self.l3(xcv,1,vlist)
        x = self.l4(x,0,vlist)
        xcv = self.l4(xcv,1,vlist)
        x = self.l5(x,0,vlist)
        xcv = self.l5(xcv,1,vlist)
        x = self.att2(x)
        x = self.l6(x,0,vlist)
        xcv = self.l6(xcv,1,vlist)
        x = self.l7(x,0,vlist)
        xcv = self.l7(xcv,1,vlist)
        x = self.l8(x,0,vlist)
        xcv = self.l8(xcv,1,vlist)
        x = self.att3(x)
        x = self.l9(x,0,vlist)
        xcv = self.l9(xcv,1,vlist)
        x = self.l10(x,0,vlist)
        xcv = self.l10(xcv,1,vlist)

        # original version, N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        # the first view-term training stage      
        c_new1 = xcv.size(1)
        xcv = xcv.view(N, M, c_new1, -1)
        xcv = xcv.mean(3).mean(1)
        xcv = self.drop_out(xcv)
        tempcv = xcv # view-common feature
        tempv = x-xcv # view-specific feature   


       
        outv = self.relu1(self.bn1(tempv))
        outv = self.fc(outv) # same-action different-view view-specific feature
        outp = self.relu1(self.bn1(tempcv))
        outp = self.fc(outp) # same-action different-view view-common feature or different-action view-common feature
        fv = self.relu(self.bn(outv))
        fv = self.prototypev(fv) # view discrimination vector based on outv


        return outv,outp,fv



        
class CTRGCN(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True):
        super(CTRGCN, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(150)

        self.encoderp_s = Model(num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True)

        bn_init(self.data_bn, 1)
        #self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

    def forward(self, x, viewt):

        # xr = _transform_rotation(x) # assistant view
        x = _transform_shape(x)
        N, C, T, V, M = x.size()
        xjmo = torch.zeros(N, C, T, V, M).cuda().half() # joint motion
        # xjmor = torch.zeros(N, C, T, V, M).cuda().half() # assistant-view joint motion
        for i in range(T-1):
            xjmo[:,:,i,:,:] = x[:,:,i+1,:,:] - x[:,:,i,:,:]
            # xjmor[:,:,i,:,:] = xr[:,:,i+1,:,:] - xr[:,:,i,:,:]

      
        outv, outp, fv = self.encoderp_s(xjmo) # 
        # outp, outp1, fp, fp1 = self.encoderp_s(xjmo, xjmor) # 

        return outv, outp, fv


# 
class MyCTRGCN(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True):
        super(MyCTRGCN, self).__init__()
        self.net = CTRGCN(num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True)

        self.net = torch.nn.DataParallel(self.net)
        #self.net = torch.nn.parallel.DistributedDataParallel(self.net) # model distributed training setting
               
    def forward(self, x, viewt):
        return self.net(x, viewt)




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