import torch
import torch.nn as nn
import math
import numpy as np
from numpy import random
from torch.autograd import Variable


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() * -(math.log(10000.0) / channel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2)]
        return x


class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=3, num_node=25, num_frame=100,
                 kernel_size=1, stride=1, glo_reg_s=True, att_s=True, glo_reg_t=True, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, use_pes=True, use_pet=True):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node, requires_grad=True)

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        self.use_temporal_att = use_temporal_att
        if use_temporal_att:
            attt = torch.zeros((1, num_subset, num_frame, num_frame))
            self.register_buffer('attt', attt)
            self.pet = PositionalEncoding(out_channels, num_node, num_frame, 'temporal')
            self.ff_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_t:
                self.in_nett = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_t:
                self.attention0t = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame), requires_grad=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            if use_temporal_att:
                self.downt1 = lambda x: x
            self.downt2 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                attention = attention + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous().view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)
            y = self.relu(self.downs1(x) + y)
            y = self.ff_nets(y)
            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)

        if self.use_temporal_att:
            attention = self.attt
            if self.use_pet:
                z = self.pet(y)
            else:
                z = y
            if self.att_t:
                q, k = torch.chunk(self.in_nett(z).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                attention = attention + self.tan(torch.einsum('nsctv,nscqv->nstq', [q, k]) / (self.inter_channels * V)) * self.alphat
            if self.glo_reg_t:
                attention = attention + self.attention0t.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            z = torch.einsum('nctv,nstq->nscqv', [y, attention]).contiguous().view(N, self.num_subset * self.out_channels, T, V)
            z = self.out_nett(z)
            z = self.relu(self.downt1(y) + z)
            z = self.ff_nett(z)
            z = self.relu(self.downt2(y) + z)
        else:
            z = self.out_nett(y)
            z = self.relu(self.downt2(y) + z)

        return z
        
class STAttentionRecBlockt(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=3, num_node=25, num_frame=100,
                 kernel_size=1, stride=1, glo_reg_s=True, att_s=True, glo_reg_t=True, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, use_pes=True, use_pet=True):
        super(STAttentionRecBlockt, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet
        self.num_frame = num_frame

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            atts1 = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            self.register_buffer('atts1', atts1)
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.in_nets2 = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.in_nets3 = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
                self.alphas1 = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node, requires_grad=True)
                self.attention0s1 = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node, requires_grad=True)

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.out_nets1 = nn.Sequential(
                nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        self.use_temporal_att = use_temporal_att
        if use_temporal_att:
            attt = torch.zeros((1, num_subset, num_frame, num_frame))
            attt1 = torch.zeros((1, num_subset, num_frame, num_frame))
            self.register_buffer('attt', attt)
            self.register_buffer('attt1', attt1)
            self.pet = PositionalEncoding(out_channels, num_node, num_frame, 'temporal')
            self.ff_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_t:
                self.in_nett = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.in_nett2 = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.in_nett3 = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
                self.alphat1 = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_t:
                self.attention0t = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame), requires_grad=True)
                self.attention0t1 = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame), requires_grad=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.out_nett1 = nn.Sequential(
                nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
                self.downs11 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downs21 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
                self.downt11 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downt22 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downt21 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
                self.downs11 = lambda x: x
            self.downs2 = lambda x: x
            self.downs21 = lambda x: x
            if use_temporal_att:
                self.downt1 = lambda x: x
                self.downt11 = lambda x: x
            self.downt2 = lambda x: x
            self.downt21 = lambda x: x
            self.downt22 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x, x1, case):

        N, C, T, V = x.size()
        
        if case == 0:
            if self.use_spatial_att:
                attention = self.atts
                attentionn = self.atts
                if self.use_pes:
                    y = self.pes(x)
                    yy = self.pes(x1)
                else:
                    y = x
                if self.att_s:
                    q1, k1 = torch.chunk(self.in_nets(y.contiguous().view(2, 2, C, T, V)[0,:,:,:,:].contiguous().view(2,C,T,V)).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                    q2, k2 = torch.chunk(self.in_nets(y.contiguous().view(2, 2, C, T, V)[1,:,:,:,:].contiguous().view(2,C,T,V)).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                    attention1 = attention + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q1, k1]) / (self.inter_channels * T)) * self.alphas
                    attention2 = attention + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q2, k2]) / (self.inter_channels * T)) * self.alphas

                    q11, k11 = torch.chunk(self.in_nets(yy.contiguous().view(2, 2, C, T, V)[0,:,:,:,:].contiguous().view(2,C,T,V)).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                    q22, k22 = torch.chunk(self.in_nets(yy.contiguous().view(2, 2, C, T, V)[1,:,:,:,:].contiguous().view(2,C,T,V)).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                    attention11 = attentionn + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q11, k11]) / (self.inter_channels * T)) * self.alphas
                    attention22 = attentionn + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q22, k22]) / (self.inter_channels * T)) * self.alphas
                    
                if self.glo_reg_s:
                    attention1 = attention1 + self.attention0s.repeat(2, 1, 1, 1)
                    attention2 = attention2 + self.attention0s.repeat(2, 1, 1, 1)

                    attention11 = attention11 + self.attention0s.repeat(2, 1, 1, 1)
                    attention22 = attention22 + self.attention0s.repeat(2, 1, 1, 1)

                attention1 = self.drop(attention1)
                attention2 = self.drop(attention2)

                attention11 = self.drop(attention11)
                attention22 = self.drop(attention22)

                y1 = torch.einsum('nctu,nsuv->nsctv', [x.contiguous().view(2, 2, C, T, V)[0,:,:,:,:].contiguous().view(2,C,T,V), attention1]).contiguous().view(2, self.num_subset * self.in_channels, T, V)
                y2 = torch.einsum('nctu,nsuv->nsctv', [x.contiguous().view(2, 2, C, T, V)[1,:,:,:,:].contiguous().view(2,C,T,V), attention2]).contiguous().view(2, self.num_subset * self.in_channels, T, V)

                y11 = torch.einsum('nctu,nsuv->nsctv', [x1.contiguous().view(2, 2, C, T, V)[0,:,:,:,:].contiguous().view(2,C,T,V), attention11]).contiguous().view(2, self.num_subset * self.in_channels, T, V)
                y22 = torch.einsum('nctu,nsuv->nsctv', [x1.contiguous().view(2, 2, C, T, V)[1,:,:,:,:].contiguous().view(2,C,T,V), attention22]).contiguous().view(2, self.num_subset * self.in_channels, T, V)

                y1 = self.out_nets(y1)
                y2 = self.out_nets(y2)
    
                y1 = self.relu(self.downs1(x.contiguous().view(2, 2, C, T, V)[0,:,:,:,:].contiguous().view(2,C,T,V)) + y1)
                y2 = self.relu(self.downs1(x.contiguous().view(2, 2, C, T, V)[1,:,:,:,:].contiguous().view(2,C,T,V)) + y2)

                y11 = self.out_nets(y11)
                y22 = self.out_nets(y22)

                y11 = self.relu(self.downs1(x1.contiguous().view(2, 2, C, T, V)[0,:,:,:,:].contiguous().view(2,C,T,V)) + y11)
                y22 = self.relu(self.downs1(x1.contiguous().view(2, 2, C, T, V)[1,:,:,:,:].contiguous().view(2,C,T,V)) + y22)
                
                q3, k3 = torch.chunk(self.in_nets2(y1).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                q4, k4 = torch.chunk(self.in_nets3(y2).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)

                q33, k33 = torch.chunk(self.in_nets2(y11).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                q44, k44 = torch.chunk(self.in_nets3(y22).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                
                attention31 = self.atts1
                attentionm = attention31 + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q33, k44]) / (self.inter_channels * T)) * self.alphas1
                attentiona = attention31 + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q44, k33]) / (self.inter_channels * T)) * self.alphas1
                attention3 = attentionm * attentiona + self.attention0s1.repeat(2, 1, 1, 1)
                attention3 = self.drop(attention3)
                y3 = torch.einsum('nctu,nsuv->nsctv', [y2, attention3]).contiguous().view(2, self.num_subset * self.out_channels, T, V)
                y3 = self.out_nets1(y3)
                y3 = self.relu(self.downs11(y3) + y3)
    
                y2 = self.ff_nets(y2)
                y3 = self.ff_nets(y3)
                y11 = self.ff_nets(y11)
                y22 = self.ff_nets(y22)
    
                y2 = self.relu(self.downs2(x.contiguous().view(2, 2, C, T, V)[1,:,:,:,:].contiguous().view(2,C,T,V)) + y2)
                y11 = self.relu(self.downs2(x1.contiguous().view(2, 2, C, T, V)[0,:,:,:,:].contiguous().view(2,C,T,V)) + y11)
                y22 = self.relu(self.downs2(x1.contiguous().view(2, 2, C, T, V)[1,:,:,:,:].contiguous().view(2,C,T,V)) + y22)
                y3 = self.relu(self.downs21(y3) + y3)
            else:
                y = self.out_nets(x)
                y = self.relu(self.downs2(x) + y)
    
            if self.use_temporal_att:
                attention = self.attt
                attention31 = self.attt1
                attentionn = self.attt

                if self.use_pet:
                    z2 = self.pet(y2)
                    z3 = self.pet(y3)
                    z22 = self.pet(y11)
                    z33 = self.pet(y22)
                else:
                    z2 = y2
                    z3 = y3
                if self.att_t:
                    q1, k1 = torch.chunk(self.in_nett(z2).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                    q2, k2 = torch.chunk(self.in_nett(z3).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)

                    attention1 = attention + self.tan(torch.einsum('nsctv,nscqv->nstq', [q1, k1]) / (self.inter_channels * V)) * self.alphat
                    attention2 = attention + self.tan(torch.einsum('nsctv,nscqv->nstq', [q2, k2]) / (self.inter_channels * V)) * self.alphat

                    q11, k11 = torch.chunk(self.in_nett(z22).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                    q22, k22 = torch.chunk(self.in_nett(z33).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)

                    attention11 = attentionn + self.tan(torch.einsum('nsctv,nscqv->nstq', [q11, k11]) / (self.inter_channels * V)) * self.alphat
                    attention22 = attentionn + self.tan(torch.einsum('nsctv,nscqv->nstq', [q22, k22]) / (self.inter_channels * V)) * self.alphat
                
                if self.glo_reg_t:
                    
                    attention1 = attention1 + self.attention0t.repeat(2, 1, 1, 1)
                    attention2 = attention2 + self.attention0t.repeat(2, 1, 1, 1)

                    attention11 = attention11 + self.attention0t.repeat(2, 1, 1, 1)
                    attention22 = attention22 + self.attention0t.repeat(2, 1, 1, 1)
                    
                attention1 = self.drop(attention1)
                attention2 = self.drop(attention2)
                attention11 = self.drop(attention11)
                attention22 = self.drop(attention22)
                
                z2 = torch.einsum('nctv,nstq->nscqv', [y2, attention1]).contiguous().view(2, self.num_subset * self.out_channels, T, V)
                z3 = torch.einsum('nctv,nstq->nscqv', [y3, attention2]).contiguous().view(2, self.num_subset * self.out_channels, T, V)
                    
                z2 = self.out_nett(z2)
                z3 = self.out_nett(z3)
    
                z2 = self.relu(self.downt1(y2) + z2)
                z3 = self.relu(self.downt1(y3) + z3)

                z22 = torch.einsum('nctv,nstq->nscqv', [y11, attention11]).contiguous().view(2, self.num_subset * self.out_channels, T, V)
                z33 = torch.einsum('nctv,nstq->nscqv', [y22, attention22]).contiguous().view(2, self.num_subset * self.out_channels, T, V)

                z22 = self.out_nett(z22)
                z33 = self.out_nett(z33)

                z22 = self.relu(self.downt1(y11) + z22)
                z33 = self.relu(self.downt1(y22) + z33)
                
                q3, k3 = torch.chunk(self.in_nett2(z2).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                q4, k4 = torch.chunk(self.in_nett3(z3).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)

                q33, k33 = torch.chunk(self.in_nett2(z22).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                q44, k44 = torch.chunk(self.in_nett3(z33).view(2, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)

                attentionn = attention31 + self.tan(torch.einsum('nsctv,nscqv->nstq', [q44, k33]) / (self.inter_channels * V)) * self.alphat1
                attentiona = attention31 + self.tan(torch.einsum('nsctv,nscqv->nstq', [q33, k44]) / (self.inter_channels * V)) * self.alphat1
                attention3 = self.drop(attentionn * attentiona)
                final = torch.einsum('nctv,nstq->nscqv', [z2, attention3]).contiguous().view(2, self.num_subset * self.out_channels, T, V)
                final = self.out_nett1(final)
                final = self.relu(self.downt11(final) + final)

                final = self.ff_nett(final)
                z2 = self.ff_nett(z2)
                z22 = self.ff_nett(z22)
                z33 = self.ff_nett(z33)
                final = self.relu(self.downt22(final) + final)
                z2 = self.relu(self.downt21(y2) + z2)
                z22 = self.relu(self.downt21(y11) + z22)
                z33 = self.relu(self.downt21(y22) + z33)
        
            else:
                z = self.out_nett(y)
                z = self.relu(self.downt2(y) + z)

            return torch.stack((final,z2),dim=0).contiguous().view(2*2,self.out_channels,final.size(2),25),torch.stack((z22,z33),dim=0).contiguous().view(2*2,self.out_channels,final.size(2),25)
        
        else:
            if self.use_spatial_att:
                attention = self.atts
                if self.use_pes:
                    y = self.pes(x)
                else:
                    y = x
                if self.att_s:
                    q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                    attention = attention + self.tan(
                    torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
                if self.glo_reg_s:
                    attention = attention + self.attention0s.repeat(N, 1, 1, 1)
                attention = self.drop(attention)
                y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
                y = self.out_nets(y)  # nctv

                y = self.relu(self.downs1(x) + y)

                y = self.ff_nets(y)

                y = self.relu(self.downs2(x) + y)
            else:
                y = self.out_nets(x)
                y = self.relu(self.downs2(x) + y)

            if self.use_temporal_att:
                attention = self.attt
                if self.use_pet:
                    z = self.pet(y)
                else:
                    z = y
                if self.att_t:
                    q, k = torch.chunk(self.in_nett(z).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                    attention = attention + self.tan(
                    torch.einsum('nsctv,nscqv->nstq', [q, k]) / (self.inter_channels * V)) * self.alphat
                if self.glo_reg_t:
                    attention = attention + self.attention0t.repeat(N, 1, 1, 1)
                attention = self.drop(attention)
                z = torch.einsum('nctv,nstq->nscqv', [y, attention]).contiguous() \
                .view(N, self.num_subset * self.out_channels, T, V)
                z = self.out_nett(z)  # nctv

                z = self.relu(self.downt1(y) + z)

                z = self.ff_nett(z)

                z = self.relu(self.downt2(y) + z)
            else:
                z = self.out_nett(y)
                z = self.relu(self.downt2(y) + z)

            return z
        
        
class STAttentionRecBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=3, num_node=25, num_frame=25,
                 kernel_size=1, stride=1, glo_reg_s=True, att_s=True, glo_reg_t=True, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, use_pes=True, use_pet=True):
        super(STAttentionRecBlocks, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                                requires_grad=True)

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        self.use_temporal_att = use_temporal_att
        if use_temporal_att:
            attt = torch.zeros((1, num_subset, num_frame, num_frame))
            self.register_buffer('attt', attt)
            self.pet = PositionalEncoding(out_channels, num_node, num_frame, 'temporal')
            self.ff_nett = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, (2, 1), (stride, 1), padding=(0, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_t:
                self.in_nett = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_t:
                self.attention0t = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame),
                                                requires_grad=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt21 = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, (2, 1), (stride, 1), padding=(0, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            if use_temporal_att:
                self.downt1 = lambda x: x
            self.downt2 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)  # nctv

            y = self.relu(self.downs1(x) + y)

            y = self.ff_nets(y)

            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)

        if self.use_temporal_att:
            attention = self.attt
            if self.use_pet:
                z = self.pet(y)
            else:
                z = y
            if self.att_t:
                q, k = torch.chunk(self.in_nett(z).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctv,nscqv->nstq', [q, k]) / (self.inter_channels * V)) * self.alphat
            if self.glo_reg_t:
                attention = attention + self.attention0t.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            z = torch.einsum('nctv,nstq->nscqv', [y, attention]).contiguous() \
                .view(N, self.num_subset * self.out_channels, T, V)
            z = self.out_nett(z)  # nctv

            z = self.relu(self.downt1(y) + z)

            z = self.ff_nett(z)

            z = self.relu(self.downt21(y) + z)
        else:
            z = self.out_nett(y)
            z = self.relu(self.downt2(y) + z)

        return z


class DSTANet(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_frame=100, num_frame1=100, num_frame2=25, num_subset=3, dropout=0., config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],], num_person=2,
                 num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True):
        super(DSTANet, self).__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]

        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.output_map = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),)

        param = {
            'num_node': num_point,
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop}
        '''
        self.graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            self.graph_layers.append(STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame, **param))
            num_frame = int(num_frame / stride + 0.5)
        '''
        self.graph_layers1t = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            self.graph_layers1t.append(
                STAttentionRecBlockt(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame1,
                                 **param))
            num_frame1 = int(num_frame1 / stride + 0.5)
        '''
        self.graph_layers1s = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate([[256, 128, 64, 2], [128, 64, 32, 2],]):
            self.graph_layers1s.append(STAttentionRecBlocks(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame2, **param))
            num_frame2 = int(num_frame2 * stride)
        '''

        self.fc = nn.Linear(self.out_channels, num_class)
        self.fc1 = nn.Linear(self.out_channels, 128)
        self.projector = nn.Linear(128, 100)

        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x, viewt):

        xr = _transform_rotation(x)
        x = _transform_shape(x)
        #xm = _transform_shape(xm)

        N, C, T, V, M = x.shape

        xbone = torch.zeros(N, C, T, V, M).cuda().half()
        xjmo = torch.zeros(N, C, T, V, M).cuda().half()
        xbmo = torch.zeros(N, C, T, V, M).cuda().half()
        xjmo1 = torch.zeros(N, C, T, V, M).cuda().half()

        pairs = ((1,2),(2,21),(3,21),(4,3),(5,21),(6,5),(7,6),(8,7),(9,21),(10,9),(11,10),(12,11),(13,1),(14,13),(15,14),(16,15),(17,1),(18,17),(19,18),(20,19),(22,23),(21,21),(23,8),(24,25),(25,12))
        for v1,v2 in pairs:
            v1-=1
            v2-=1
            xbone[:,:,:,v1,:] = x[:,:,:,v1,:] - x[:,:,:,v2,:]

        for i in range(T-1):
            xjmo[:,:,i,:,:] = x[:,:,i+1,:,:] - x[:,:,i,:,:]
            xjmo1[:,:,i,:,:] = xr[:,:,i+1,:,:] - xr[:,:,i,:,:]
            xbmo[:,:,i,:,:] = xbone[:,:,i+1,:,:] - xbone[:,:,i,:,:]

        xmo = torch.cat((xjmo,xbmo),dim=0)

        temp = x
        tempr = xr
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        xr = xr.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x = self.input_map(x)
        xr = self.input_map(xr)

        for i, m in enumerate(self.graph_layers1t):
            x = m(x,x,1)
            xr = m(xr,xr,1)

        '''
        xm = _transform_shape(xm)
        if case == 0:
            xm1 = xm
        else:        
            count = 0
            for i in range(N):
                if (viewtm[i]==1)|(viewtm[i]==7):
                    count+=1
            xm1 = torch.zeros(count,C,T,V,M)
            count1=0
            for i in range(N):
                if (viewtm[i]==1)|(viewtm[i]==7):
                    xm1[count1,:,:,:,:]=xm[i,:,:,:,:]
                    count1+=1
        
        count = xm1.size(0)
        xm1 = xm1.permute(0, 4, 1, 2, 3).contiguous().view(count * M, C, T, V).cuda().half()
        xm1 = self.input_map(xm1)        
        for i, m in enumerate(self.graph_layers):
            xm1 = m(xm1)
        xm2 = xm1        
        for i, m in enumerate(self.graph_layers1s):
            xm1 = m(xm1)
        xm1 = self.output_map(xm1)
        xm1 = self.input_map(xm1)
        for i, m in enumerate(self.graph_layers):
            xm1 = m(xm1)
        
        xm1 = xm1.view(count, M, self.out_channels, -1)
        xm1 = xm1.permute(0, 1, 3, 2).contiguous().view(count, -1, self.out_channels, 1)  # whole channels of one spatial
        xm1 = self.drop_out2d(xm1)
        xm1 = xm1.mean(3).mean(1)
        xm1 = self.drop_out(xm1)  # whole spatial of one channel
        xm1 = self.fc(xm1)
               
        xm2 = xm2.view(count, M, self.out_channels, -1)
        xm2 = xm2.permute(0, 1, 3, 2).contiguous().view(count, -1, self.out_channels, 1)  # whole channels of one spatial
        xm2 = self.drop_out2d(xm2)
        xm2 = xm2.mean(3).mean(1)
        xm2 = self.drop_out(xm2)  # whole spatial of one channel
        xm2 = self.fc(xm2)
        
        preg1 = xm2
        #preg2 = xm1
        temps = x
        temp = recg
        temp = temp.contiguous().view(N, M, 3, 100, V)
        tempr = xr
        tempr = tempr.contiguous().view(N, M, 256, -1, V)

        temps = x
        for i, m in enumerate(self.graph_layers1s):
            temps = m(temps)
        temps = self.output_map(temps)
        rec1 = temps
        temps = self.input_map(temps)
        for i, m in enumerate(self.graph_layers):
            temps = m(temps)
        '''

        '''
        list0,list1,list2,list3,list4,list5,list6,list7 = list(),list(),list(),list(),list(),list(),list(),list()
        
        for i in range(N):          
            if viewt[i] == 0:
                list0.append(x[i,:,:,:,:].view(1,2,C,T,V))
            if viewt[i] == 1:
                list1.append(x[i,:,:,:,:].view(1,2,C,T,V))
            if viewt[i] == 2:
                list2.append(x[i,:,:,:,:].view(1,2,C,T,V))
            if viewt[i] == 3:
                list3.append(x[i,:,:,:,:].view(1,2,C,T,V))
            if viewt[i] == 4:
                list4.append(x[i,:,:,:,:].view(1,2,C,T,V))
            if viewt[i] == 5:
                list5.append(x[i,:,:,:,:].view(1,2,C,T,V))
            if viewt[i] == 6:
                list6.append(x[i,:,:,:,:].view(1,2,C,T,V))
            if viewt[i] == 7:
                list7.append(x[i,:,:,:,:].view(1,2,C,T,V))
            
            if viewt[i] == 1:
                list1.append(temp[i,:,:,:,:].view(1,2,3,100,V))
            if viewt[i] == 3:
                list3.append(temp[i,:,:,:,:].view(1,2,3,100,V))
            if viewt[i] == 5:
                list5.append(temp[i,:,:,:,:].view(1,2,3,100,V))
            if viewt[i] == 7:
                list7.append(temp[i,:,:,:,:].view(1,2,3,100,V))
        
        teacher = torch.zeros((N,2,256,25,V)).half().cuda()
        teacher1 = torch.zeros((N,2,256,25,V)).half().cuda()
        
            choice = random.randint(0,5)
            if choice == 0:
                if (len(list1) != 0)&(len(list7) != 0):
                    vmain = random.choice(list1)
                    vassist = random.choice(list7)
                if (len(list1) == 0)&(len(list7) != 0):
                    vmain = random.choice(list7)
                    vassist = random.choice(list7)
                if (len(list7) == 0)&(len(list1) != 0): 
                    vmain = random.choice(list1)
                    vassist = random.choice(list1)
                if (len(list1) == 0)&(len(list7) == 0):
                    choice = 1

            if choice == 1:
                if (len(list1) != 0)&(len(list5) != 0):
                    vmain = random.choice(list1)
                    vassist = random.choice(list5)
                if (len(list1) == 0)&(len(list5) != 0):
                    vmain = random.choice(list5)
                    vassist = random.choice(list5)
                if (len(list5) == 0)&(len(list1) != 0):
                    vmain = random.choice(list1)
                    vassist = random.choice(list1)
                if (len(list1) == 0)&(len(list5) == 0):
                    choice = 2

            if choice == 2:
                if (len(list1) != 0)&(len(list3) != 0):
                    vmain = random.choice(list1)
                    vassist = random.choice(list3)
                if (len(list1) == 0)&(len(list3) != 0):
                    vmain = random.choice(list3)
                    vassist = random.choice(list3)
                if (len(list3) == 0)&(len(list1) != 0):
                    vmain = random.choice(list1)
                    vassist = random.choice(list1)
                if (len(list1) == 0)&(len(list3) == 0):
                    choice = 3

            if choice == 3:
                if (len(list3) != 0)&(len(list7) != 0):
                    vmain = random.choice(list3)
                    vassist = random.choice(list7)
                if (len(list3) == 0)&(len(list7) != 0):
                    vmain = random.choice(list7)
                    vassist = random.choice(list7)
                if (len(list7) == 0)&(len(list3) != 0):
                    vmain = random.choice(list3)
                    vassist = random.choice(list3)
                if (len(list3) == 0)&(len(list7) == 0):
                    choice = 4

            if choice == 4:
                if (len(list3) != 0)&(len(list5) != 0):
                    vmain = random.choice(list3)
                    vassist = random.choice(list5)
                if (len(list3) == 0)&(len(list5) != 0):
                    vmain = random.choice(list5)
                    vassist = random.choice(list5)
                if (len(list5) == 0)&(len(list3) != 0):
                    vmain = random.choice(list3)
                    vassist = random.choice(list3)
                if (len(list3) == 0)&(len(list5) == 0):
                    choice = 5

            if choice == 5:
                if (len(list5) != 0)&(len(list7) != 0):
                    vmain = random.choice(list5)
                    vassist = random.choice(list7)
                if (len(list5) == 0)&(len(list7) != 0):
                    vmain = random.choice(list7)
                    vassist = random.choice(list7)
                if (len(list7) == 0)&(len(list5) != 0):
                    vmain = random.choice(list5)
                    vassist = random.choice(list5)
                if (len(list5) == 0)&(len(list7) == 0):
                    choice = 0
            '''

        teacher = torch.zeros((N,2,256,25,V)).half().cuda()
        teacher1 = torch.zeros((N,2,256,25,V)).half().cuda()

        for i in range(N):
            #choice = random.randint(0,N-1)
            temp1 = torch.cat((temp[i,:,:,:,:].view(1,2,3,100,V),tempr[i,:,:,:,:].view(1,2,3,100,V)),dim=0).contiguous().view(2*2, 3, 100, V)
            temp2 = torch.cat((xjmo[i,:,:,:,:].view(1,2,3,100,V),xjmo1[i,:,:,:,:].view(1,2,3,100,V)),dim=0).contiguous().view(2*2, 3, 100, V)
            temp3 = torch.cat((tempr[i,:,:,:,:].view(1,2,3,100,V),temp[i,:,:,:,:].view(1,2,3,100,V)),dim=0).contiguous().view(2*2, 3, 100, V)
            temp4 = torch.cat((xjmo1[i,:,:,:,:].view(1,2,3,100,V),xjmo[i,:,:,:,:].view(1,2,3,100,V)),dim=0).contiguous().view(2*2, 3, 100, V)

            temp1 = self.input_map(temp1)
            temp2 = self.input_map(temp2)
            temp3 = temp1
            temp4 = temp2
            for i, m in enumerate(self.graph_layers1t):
                temp1, temp2 = m(temp1,temp2,0)
                temp3, temp4 = m(temp3,temp4,0)

            teacher[i,:,:,:,:] = temp1.contiguous().view(2, 2, 256, 25, V)[0,:,:,:,:]
            teacher1[i,:,:,:,:] = temp3.contiguous().view(2, 2, 256, 25, V)[0,:,:,:,:]

        '''
        teacher = teacher.contiguous().view(N*M, 64, T, V)
        teacher = self.output_map(teacher)
        rec2 = teacher
        teacher = self.input_map(teacher)
        for i, m in enumerate(self.graph_layers):
            teacher = m(teacher)
            
        teacher1 = teacher1.contiguous().view(N*M, 64, T, V)
        teacher1 = self.output_map(teacher1)
        rec3 = teacher1
        teacher1 = self.input_map(teacher1)
        for i, m in enumerate(self.graph_layers):
            teacher1 = m(teacher1)
        '''

        '''
        temps = temps.view(N, M, self.out_channels, -1)
        temps = temps.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
        temps = self.drop_out2d(temps)
        temps = temps.mean(3).mean(1)
        temps = self.drop_out(temps)  # whole spatial of one channel
        temps = self.fc(temps)
        pre1 = temps
        
        xm = xm.view(N, M, self.out_channels, -1)
        xm = xm.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
        xm = self.drop_out2d(xm)
        xm = xm.mean(3).mean(1)
        xm = self.drop_out(xm)  # whole spatial of one channel
        xm = self.fc(xm)
        pre2 = xm

        xm1 = xm1.view(N, M, self.out_channels, -1)
        xm1 = xm1.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
        xm1 = self.drop_out2d(xm1)
        xm1 = xm1.mean(3).mean(1)
        xm1 = self.drop_out(xm1)  # whole spatial of one channel
        xm1 = self.fc(xm1)
        pre5 = xm1       
        '''

        teacher = teacher.view(N, M, self.out_channels, -1)
        teacher = teacher.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
        teacher = self.drop_out2d(teacher)
        teacher = teacher.mean(3).mean(1)
        teacher = self.drop_out(teacher)
        fctemp = teacher
        fc = self.fc1(teacher)
        fc = self.projector(fc)

        teacher1 = teacher1.view(N, M, self.out_channels, -1)
        teacher1 = teacher1.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
        teacher1 = self.drop_out2d(teacher1)
        teacher1 = teacher1.mean(3).mean(1)
        teacher1 = self.drop_out(teacher1)
        fctemp1 = teacher1
        fc1 = self.fc1(teacher1)
        fc1 = self.projector(fc1)
        
        '''
        x = x.view(N, M, self.out_channels, -1)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)  # whole spatial of one channel
        fstemp = x-fctemp
        fs = self.fc1(fstemp)
        '''

        return fc,fc1

        
class DSTANet1(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_frame=100, num_frame1=25, num_frame2=25, num_subset=3, dropout=0., config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],], num_person=2,
                 num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True):
        super(DSTANet1, self).__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]

        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.output_map = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),)

        param = {
            'num_node': num_point,
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop}

        self.graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            self.graph_layers.append(STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame, **param))
            num_frame = int(num_frame / stride + 0.5)
            
        self.graph_layers1t = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate([[256, 128, 64, 2], [128, 64, 32, 2],]):
            self.graph_layers1t.append(STAttentionRecBlockt(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame1, **param))
            num_frame1 = int(num_frame1 * stride)
        
        self.graph_layers1s = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate([[256, 128, 64, 2], [128, 64, 32, 2],]):
            self.graph_layers1s.append(STAttentionRecBlocks(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame2, **param))
            num_frame2 = int(num_frame2 * stride)

        self.fc = nn.Linear(self.out_channels, num_class)
        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):

        x = _transform_shape(x)
        N, C, T, V, M = x.shape

        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        recg = x
        x = self.input_map(x)

        for i, m in enumerate(self.graph_layers):
            x = m(x)
        
        x = x.view(N, M, self.out_channels, -1)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc(x)
        
        return x
        

class MyDSTANet(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_frame=100, num_frame1=100, num_frame2=25, num_subset=3, dropout=0., config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],], num_person=2,
                 num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True):
        super(MyDSTANet, self).__init__()

        self.net = DSTANet(num_class=40, num_point=25, num_frame=100, num_frame1=100, num_frame2=25, num_subset=3, dropout=0., config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],], num_person=2,
                 num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True)
        self.net = torch.nn.DataParallel(self.net)
    
    def forward(self, x, viewt):
        
        return self.net(x, viewt)


class MyDSTANet1(nn.Module):
    def __init__(self, num_class=40, num_point=25, num_frame=100, num_frame1=25, num_frame2=25, num_subset=3, dropout=0., config = [[64, 64, 16, 1], [64, 64, 16, 1],
                                                                                                                                    [64, 128, 32, 2], [128, 128, 32, 1],
                                                                                                                                    [128, 256, 64, 2], [256, 256, 64, 1],
                                                                                                                                    [256, 256, 64, 1], [256, 256, 64, 1],], num_person=2,
                 num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True):
        super(MyDSTANet1, self).__init__()

        self.net = DSTANet1(num_class=40, num_point=25, num_frame=100, num_frame1=25, num_frame2=25, num_subset=3, dropout=0., config = [[64, 64, 16, 1], [64, 64, 16, 1],
                                                                                                                                        [64, 128, 32, 2], [128, 128, 32, 1],
                                                                                                                                        [128, 256, 64, 2], [256, 256, 64, 1],                                                                                                                                       [256, 256, 64, 1], [256, 256, 64, 1],], num_person=2,
                           num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True,
                           use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True)
        self.net = torch.nn.DataParallel(self.net)

    def forward(self, x, xm, viewt):

        return self.net(x, xm, viewt)

        
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
    inputs2 = torch.cat((inputs2, Variable(inputs2.data.new(inputs2.size()[0], 1, inputs2.size()[2]).fill_(1))), dim=1).cuda()
    inputs2 = torch.matmul(trans.half(), inputs2)
    inputs2 = torch.matmul(rot.half(), inputs2)
    inputs2 = inputs2.contiguous().view(-1, 3, 3014, 25)
    inputs2 = torch.split(inputs2, [300,2714], dim=2)[0]
    inputs2 = torch.index_select(inputs2, 2, torch.arange(1,300,3).cuda())

    inputs = torch.stack([inputs1, inputs2], dim=4)

    return inputs

'''
if __name__ == '__main__':
    config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],
              ]
    net = DSTANet(config=config)  # .cuda()
    ske = torch.rand([2, 3, 32, 25, 2])  # .cuda()
    print(net(ske).shape)
'''