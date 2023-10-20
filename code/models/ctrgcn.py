import math
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from numpy import random
from graph.ntu_rgb_d import AdjMatrixGraph, AdjMatrixGraph4lh, AdjMatrixGraph4rh, AdjMatrixGraph4ll, AdjMatrixGraph4rl, AdjMatrixGraph4torso
from itertools import permutations
from apex import amp, optimizers

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

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)

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

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        self.apply(weights_init)

    def forward(self, x):

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


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=False)
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
    def __init__(self, num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True, case=1):
        super(Model, self).__init__()

        Graph = AdjMatrixGraph()
        A = Graph.A_binary

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.att1 = ST_Joint_Att(base_channel, 4, True)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.att2 = ST_Joint_Att(base_channel*2, 4, True)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.att3 = ST_Joint_Att(base_channel*4, 4, True)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.case = case
        #self.fc = nn.Linear(base_channel*4, num_class)

        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, case):

        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        if case==1:
            x = self.l1(x)
            x = self.att1(x)
            x = self.l2(x)
            x = self.l3(x)
            x = self.l4(x)
            x = self.l5(x)
            x = self.att2(x)
            x = self.l6(x)
            x = self.l7(x)
            x = self.l8(x)
            x = self.att3(x)
            x = self.l9(x)
            x = self.l10(x)
        else:
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
            x = self.l4(x)
            x = self.l5(x)
            x = self.l6(x)
            x = self.l7(x)
            x = self.l8(x)
            x = self.l9(x)
            x = self.l10(x)

        return x


class CTRGCN(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True):
        super(CTRGCN, self).__init__()
        #Graph = AdjMatrixGraph()
        #A = Graph.A_binary

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(150)
        self.encoder_s = Model(num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True, case=1)
        self.encoder_t = Model(num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True)
        self.classifier = nn.Linear(256, 40)

        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2. / 40))
        bn_init(self.data_bn, 1)


    def forward(self, x, xm, viewt, t):
        
        viewt = list(viewt)
        t = list(t)
        #xv = _transform_rotation(xm)
        x = _transform_shape(x)
        xm = _transform_shape(xm)
        N, C, T, V, M = x.size()
        xbone = torch.zeros(N, C, T, V, M).cuda()
        xjmo = torch.zeros(N, C, T, V, M).cuda()
        xbmo = torch.zeros(N, C, T, V, M).cuda()
        xbone1 = torch.zeros(N, C, T, V, M).cuda()
        xjmo1 = torch.zeros(N, C, T, V, M).cuda()
        xbmo1 = torch.zeros(N, C, T, V, M).cuda()

        pairs = ((1,2),(2,21),(3,21),(4,3),(5,21),(6,5),(7,6),(8,7),(9,21),(10,9),(11,10),(12,11),(13,1),(14,13),(15,14),(16,15),(17,1),(18,17),(19,18),(20,19),(22,23),(21,21),(23,8),(24,25),(25,12))
        for v1,v2 in pairs:
            v1-=1
            v2-=1
            xbone[:,:,:,v1,:] = x[:,:,:,v1,:] - x[:,:,:,v2,:]
            xbone1[:,:,:,v1,:] = xm[:,:,:,v1,:] - xm[:,:,:,v2,:]

        for i in range(T-1):
            xjmo[:,:,i,:,:] = x[:,:,i+1,:,:] - x[:,:,i,:,:]
            xbmo[:,:,i,:,:] = xbone[:,:,i+1,:,:] - xbone[:,:,i,:,:]
            xjmo1[:,:,i,:,:] = xm[:,:,i+1,:,:] - xm[:,:,i,:,:]
            xbmo1[:,:,i,:,:] = xbone1[:,:,i+1,:,:] - xbone1[:,:,i,:,:]

        return
        

class MyCTRGCN(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True):
        super(MyCTRGCN, self).__init__()
        self.net = CTRGCN(num_class=40, num_point=25, num_person=2, graph=None, in_channels=3, drop_out=0, adaptive=True)

        self.net = torch.nn.DataParallel(self.net)
        #self.net = torch.nn.parallel.DistributedDataParallel(self.net)
               
    def forward(self, x, xm, viewt, t):
        
        return self.net(x, xm, viewt, t)


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
            x[i][j] = random.choice([0,math.pi/36,math.pi/18,3*math.pi/36,math.pi/9,5*math.pi/36,6*math.pi/36,7*math.pi/36,8*math.pi/36,math.pi/4])
            #x[i][j] = 0
    y = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            #y[i][j] = random.randrange(0,2*math.pi,math.pi/8)
            viewchoice = random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
            view.append(viewchoice)
            y[i][j] = view1[viewchoice]
    
    z = np.empty([inputs1.shape[0],1], dtype = float) 
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            #z[i][j] = random.randrange(0,math.pi/4,math.pi/36)
            z[i][j] = random.choice([0,math.pi/36,math.pi/18,3*math.pi/36,math.pi/9,5*math.pi/36,6*math.pi/36,7*math.pi/36,8*math.pi/36,math.pi/4])
            #z[i][j] = 0
    rot_x = torch.tensor(x)
    rot_y = torch.tensor(y)
    rot_z = torch.tensor(z)
    rot = torch.cat((rot_x,rot_y,rot_z),dim=1).float().cuda()
    t = np.empty([inputs1.shape[0],3], dtype = float) 
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            t[i][j] = random.uniform(-0.05,0.05)
    trans = torch.tensor(t).half().cuda()
    trans, rot = _trans_rot(trans, rot)
    inputs1 = inputs1.contiguous().view(-1, inputs1.size()[1], inputs1.size()[2] * inputs1.size()[3])
    inputs1 = torch.cat((inputs1, Variable(inputs1.data.new(inputs1.size()[0], 1, inputs1.size()[2]).fill_(1))), dim=1).cuda()
    inputs1 = torch.matmul(trans.half(), inputs1)
    inputs1 = torch.matmul(rot.half(), inputs1)
    inputs1 = inputs1.contiguous().view(-1, 3, 3014, 25)
    inputs1 = torch.split(inputs1, [300,2714], dim=2)[0]
    inputs1 = torch.index_select(inputs1, 2, torch.arange(1,300,3).cuda())

    b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55 = a2.chunk(25, dim=2)
    inputs2 = torch.stack([b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55], dim=3)
    inputs2 = inputs2.permute(0, 2, 1, 3)
    inputs2 = inputs2.contiguous().view(-1, inputs2.size()[1], inputs2.size()[2] * inputs2.size()[3])
    inputs2 = torch.cat((inputs2, Variable(inputs2.data.new(inputs2.size()[0], 1, inputs2.size()[2]).fill_(1))),
                        dim=1).cuda()
    inputs2 = torch.matmul(trans.half(), inputs2)
    inputs2 = torch.matmul(rot.half(), inputs2)
    inputs2 = inputs2.contiguous().view(-1, 3, 3014, 25)
    inputs2 = torch.split(inputs2, [300,2714], dim=2)[0]
    inputs2 = torch.index_select(inputs2, 2, torch.arange(1,300,3).cuda())

    inputs = torch.stack([inputs1, inputs2], dim=4)

    return inputs


def _transform(inputs1,mat):
    
    rot = mat[:,0:3]
    trans = mat[:,3:6]
    
    a1, a2 = inputs1.split([75, 75], dim=2)
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25 = a1.chunk(
        25, dim=2)
    inputs1 = torch.stack(
        [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24,
         b25], dim=3)
    inputs1 = inputs1.permute(0, 2, 1, 3)
    
    trans, rot = _trans_rot(trans, rot)
    inputs1 = inputs1.contiguous().view(-1, inputs1.size()[1], inputs1.size()[2] * inputs1.size()[3])
    inputs1 = torch.cat((inputs1, Variable(inputs1.data.new(inputs1.size()[0], 1, inputs1.size()[2]).fill_(1))),
                        dim=1).cuda()
    inputs1 = torch.matmul(trans, inputs1)
    inputs1 = torch.matmul(rot, inputs1)
    inputs1 = inputs1.contiguous().view(-1, 3, 100, 25)
    #inputs1 = torch.split(inputs1, [500,2514], dim=2)[0]

    b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55 = a2.chunk(25, dim=2)
    inputs2 = torch.stack([b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55], dim=3)
    inputs2 = inputs2.permute(0, 2, 1, 3)
    inputs2 = inputs2.contiguous().view(-1, inputs2.size()[1], inputs2.size()[2] * inputs2.size()[3])
    inputs2 = torch.cat((inputs2, Variable(inputs2.data.new(inputs2.size()[0], 1, inputs2.size()[2]).fill_(1))), dim=1).cuda()
    inputs2 = torch.matmul(trans, inputs2)
    inputs2 = torch.matmul(rot, inputs2)
    inputs2 = inputs2.contiguous().view(-1, 3, 100, 25)
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
    inputs1 = torch.index_select(inputs1, 2, torch.arange(1,300,3).cuda())
    b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55 = a2.chunk(25, dim=2)
    inputs2 = torch.stack([b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55], dim=3)
    inputs2 = inputs2.permute(0, 2, 1, 3)
    inputs2 = inputs2.contiguous().view(-1, 3, 3014, 25)
    inputs2 = torch.split(inputs2, [300,2714], dim=2)[0]
    inputs2 = torch.index_select(inputs2, 2, torch.arange(1,300,3).cuda())
    inputs = torch.stack([inputs1, inputs2], dim=4)

    return inputs