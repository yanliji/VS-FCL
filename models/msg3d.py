import sys
sys.path.insert(0, '')
import math
import torch
import random
import torch.nn as nn
import numpy as np
from numpy import random
import torch.nn.functional as F
from models.mlp import MLP
from util import import_class, count_params
from models.ms_gcn import MultiScale_GraphConv as MS_GCN
from models.ms_tcn import MultiScale_TemporalConv as MS_TCN
from models.ms_tcn import TemporalDeConv
from models.ms_tcn import TemporalConv
from models.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from models.activation import activation_factory
from itertools import permutations
from apex import amp, optimizers
from torch.autograd import Variable
from graph.ntu_rgb_d import AdjMatrixGraph, AdjMatrixGraph4lh, AdjMatrixGraph4rh, AdjMatrixGraph4ll, AdjMatrixGraph4rl, AdjMatrixGraph4torso

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


'''
class InnerProductDecoder(nn.Module):

    def __init__(self, input_dim, dropout=0., act=nn.Sigmoid()):
        super(InnerProductDecoder, self).__init__()
        self.act = act
        self.dropout = dropout
        self.dropoutnet = nn.Dropout(1-self.dropout)

    def forward(self, inputs):
        inputs = self.dropoutnet(inputs)
        x = torch.transpose(inputs)
        x = torch.matmul(inputs, x)
        x = torch.reshape(x, [-1])
        outputs = self.act(x)
        return outputs
'''

'''
class Model4location(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model4location, self).__init__()

        # respectly define 5 sub graphs for limbs and torso
        Graph1 = AdjMatrixGraph4lh()
        A_binary1 = Graph1.A_binary
        Graph2 = AdjMatrixGraph4rh()
        A_binary2 = Graph2.A_binary
        Graph3 = AdjMatrixGraph4ll()
        A_binary3 = Graph3.A_binary
        Graph4 = AdjMatrixGraph4rl()
        A_binary4 = Graph4.A_binary
        Graph5 = AdjMatrixGraph4torso()
        A_binary5 = Graph5.A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * 6)
        self.data_bn1 = nn.BatchNorm1d(num_person * in_channels * 4)
        self.data_bn2 = nn.BatchNorm1d(num_person * in_channels * 9)

        c1 = 96
        c2 = c1 * 2     
        c3 = c2 * 2     

        # respectly define gcn for every sub graph, gcn3d and sgcn, 1/2/3 refers to A1, 11/21/31 refers to A2, 12/22/32 refers to A3 and so on 
        self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary1, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary1, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary1, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary1, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary1, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary1, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        #self.fc_final = nn.Linear(128, num_class)
        self.fc = nn.Linear(c3, 128)
        self.fc2 = nn.Linear(128, 5)
        
        self.gcn3d11 = MultiWindow_MS_G3D(3, c1, A_binary2, num_g3d_scales, window_stride=1)
        self.sgcn11 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary2, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
            
        self.gcn3d21 = MultiWindow_MS_G3D(c1, c2, A_binary2, num_g3d_scales, window_stride=2)
        self.sgcn21 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary2, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
            
        self.gcn3d31 = MultiWindow_MS_G3D(c2, c3, A_binary2, num_g3d_scales, window_stride=2)
        self.sgcn31 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary2, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
            
        self.gcn3d12 = MultiWindow_MS_G3D(3, c1, A_binary3, num_g3d_scales, window_stride=1)
        self.sgcn12 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary3, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
            
        self.gcn3d22 = MultiWindow_MS_G3D(c1, c2, A_binary3, num_g3d_scales, window_stride=2)
        self.sgcn22 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary3, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
            
        self.gcn3d32 = MultiWindow_MS_G3D(c2, c3, A_binary3, num_g3d_scales, window_stride=2)
        self.sgcn32 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary3, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
            
        self.gcn3d13 = MultiWindow_MS_G3D(3, c1, A_binary4, num_g3d_scales, window_stride=1)
        self.sgcn13 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary4, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
            
        self.gcn3d23 = MultiWindow_MS_G3D(c1, c2, A_binary4, num_g3d_scales, window_stride=2)
        self.sgcn23 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary4, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
            
        self.gcn3d33 = MultiWindow_MS_G3D(c2, c3, A_binary4, num_g3d_scales, window_stride=2)
        self.sgcn33 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary4, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
            
        self.gcn3d14 = MultiWindow_MS_G3D(3, c1, A_binary5, num_g3d_scales, window_stride=1)
        self.sgcn14 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary5, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
            
        self.gcn3d24 = MultiWindow_MS_G3D(c1, c2, A_binary5, num_g3d_scales, window_stride=2)
        self.sgcn24 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary5, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
            
        self.gcn3d34 = MultiWindow_MS_G3D(c2, c3, A_binary5, num_g3d_scales, window_stride=2)
        self.sgcn34 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary5, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        
    def forward(self, x):
    
        # there is a problem that whether the data corresponds to graph, also same with above, one batch uses same sub graph
        choice = random.randint(1,5)
        if choice == 1:
            x = torch.index_select(x, 3, torch.tensor([9,10,11,12,24,25]).cuda())
        if choice == 2:
            x = torch.index_select(x, 3, torch.tensor([5,6,7,8,22,23]).cuda())
        if choice == 3:
            x = torch.index_select(x, 3, torch.tensor([17,18,19,20]).cuda())
        if choice == 4:
            x = torch.index_select(x, 3, torch.tensor([13,14,15,16]).cuda())
        if choice == 5:
            x = torch.index_select(x, 3, torch.tensor([1,2,3,4,5,9,13,17,21]).cuda())
               
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        if (choice == 1)|(choice == 2):
            x = self.data_bn(x)
        if (choice == 3)|(choice == 4):
            x = self.data_bn1(x)
        if choice == 5:
            x = self.data_bn2(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        
        # for every choice, define respective forward process
        if choice == 1:
            x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
            x = self.tcn1(x)
            x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
            x = self.tcn2(x)
            x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
            x = self.tcn3(x)
        
        if choice == 2:
            x = F.relu(self.sgcn11(x) + self.gcn3d11(x), inplace=True)
            x = self.tcn1(x)
            x = F.relu(self.sgcn21(x) + self.gcn3d21(x), inplace=True)
            x = self.tcn2(x)
            x = F.relu(self.sgcn31(x) + self.gcn3d31(x), inplace=True)
            x = self.tcn3(x)
            
        if choice == 3:
            x = F.relu(self.sgcn12(x) + self.gcn3d12(x), inplace=True)
            x = self.tcn1(x)
            x = F.relu(self.sgcn22(x) + self.gcn3d22(x), inplace=True)
            x = self.tcn2(x)
            x = F.relu(self.sgcn32(x) + self.gcn3d32(x), inplace=True)
            x = self.tcn3(x)
            
        if choice == 4:
            x = F.relu(self.sgcn13(x) + self.gcn3d13(x), inplace=True)
            x = self.tcn1(x)
            x = F.relu(self.sgcn23(x) + self.gcn3d23(x), inplace=True)
            x = self.tcn2(x)
            x = F.relu(self.sgcn33(x) + self.gcn3d33(x), inplace=True)
            x = self.tcn3(x)
            
        if choice == 5:
            x = F.relu(self.sgcn14(x) + self.gcn3d14(x), inplace=True)
            x = self.tcn1(x)
            x = F.relu(self.sgcn24(x) + self.gcn3d24(x), inplace=True)
            x = self.tcn2(x)
            x = F.relu(self.sgcn34(x) + self.gcn3d34(x), inplace=True)
            x = self.tcn3(x)
        
        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   
        out = out.mean(1) 
        out = self.fc(out)
        out = self.fc2(out)
        
        # define label
        label = np.zeros((N,5))
        for row in range(0,N):
            label[row][choice-1] = 1
        lebel = torch.tensor(label).cuda()
              
        return out,label
'''


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
        '''
        Graph1 = AdjMatrixGraph4lh()
        A_binary1 = Graph1.A_binary
        Graph2 = AdjMatrixGraph4rh()
        A_binary2 = Graph2.A_binary
        Graph3 = AdjMatrixGraph4ll()
        A_binary3 = Graph3.A_binary
        Graph4 = AdjMatrixGraph4rl()
        A_binary4 = Graph4.A_binary
        Graph5 = AdjMatrixGraph4torso()
        A_binary5 = Graph5.A_binary
        '''

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

        '''
        self.gcn3d11 = MultiWindow_MS_G3D(3, c1, A_binary1, num_g3d_scales, window_stride=1)
        self.sgcn11 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary1, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
            
        self.gcn3d21 = MultiWindow_MS_G3D(c1, c2, A_binary1, num_g3d_scales, window_stride=2)
        self.sgcn21 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binar1, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
            
        self.gcn3d31 = MultiWindow_MS_G3D(c2, c3, A_binary1, num_g3d_scales, window_stride=2)
        self.sgcn31 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary1, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
            
        self.gcn3d12 = MultiWindow_MS_G3D(3, c1, A_binary2, num_g3d_scales, window_stride=1)
        self.sgcn12 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary2, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
            
        self.gcn3d22 = MultiWindow_MS_G3D(c1, c2, A_binary2, num_g3d_scales, window_stride=2)
        self.sgcn22 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary2, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
            
        self.gcn3d32 = MultiWindow_MS_G3D(c2, c3, A_binary2, num_g3d_scales, window_stride=2)
        self.sgcn32 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary2, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
            
        self.gcn3d13 = MultiWindow_MS_G3D(3, c1, A_binary3, num_g3d_scales, window_stride=1)
        self.sgcn13 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary3, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
            
        self.gcn3d23 = MultiWindow_MS_G3D(c1, c2, A_binary3, num_g3d_scales, window_stride=2)
        self.sgcn23 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary3, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
            
        self.gcn3d33 = MultiWindow_MS_G3D(c2, c3, A_binary3, num_g3d_scales, window_stride=2)
        self.sgcn33 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary3, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
            
        self.gcn3d14 = MultiWindow_MS_G3D(3, c1, A_binary4, num_g3d_scales, window_stride=1)
        self.sgcn14 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary4, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
            
        self.gcn3d24 = MultiWindow_MS_G3D(c1, c2, A_binary4, num_g3d_scales, window_stride=2)
        self.sgcn24 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary4, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
            
        self.gcn3d34 = MultiWindow_MS_G3D(c2, c3, A_binary4, num_g3d_scales, window_stride=2)
        self.sgcn34 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary4, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
            
        self.gcn3d15 = MultiWindow_MS_G3D(3, c1, A_binary5, num_g3d_scales, window_stride=1)
        self.sgcn15 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary5, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
            
        self.gcn3d25 = MultiWindow_MS_G3D(c1, c2, A_binary5, num_g3d_scales, window_stride=2)
        self.sgcn25 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary5, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
            
        self.gcn3d35 = MultiWindow_MS_G3D(c2, c3, A_binary5, num_g3d_scales, window_stride=2)
        self.sgcn35 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary5, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        '''

        #self.gcnr = MS_GCN(num_gcn_scales, c3, c1, A_binary, disentangled_agg=True)
        #self.tcnr = TemporalDeConv(3, 3)
        #self.reconstructions = InnerProductDecoder(input_dim=c3, act=lambda x: x)
        self.fc = nn.Linear(c3, 256)
        self.prototype = nn.Linear(c3, 1000)

    def forward(self, x, xm):
        '''
        # define lists and dictionaries
        #order_list = list(permutations([1, 2, 3, 4, 5], 5))
        #order_label = dict()
        #select_order_label = dict()
        #count = dict()
        # define the whole permutation sets
        #for idx, element in enumerate(order_list):
        #    order_label[idx] = order_list[idx]
        #select_order_label = order_label
        # define the selected permutation sets

        for i in range(0,32):
            if i == 0:
                select_order_label[i] = order_label[i]
                order_label.pop(0)
            else:
                for idx, element in order_label.items():          
                    m = 0 
                    for key, element1 in select_order_label.items():
                        m += hmd(element,element1)
                    count[idx] = m
            
                min_index, min = list(count.items())[0]
                for key,value in count.items():         
                    if value < min:
                        min = value
                        min_index = key
                select_order_label[i] = order_label[min_index]
                order_label.pop(min_index)
                count.pop(min_index)
        '''

        '''
        x1, x2, x3 = x[:,:,0:50,:,:], x[:,:,50:150,:,:], x[:,:,150:300,:,:]
        xlist = [x1,x2,x3]
        y1, y2, y3 = xv[:,:,0:50,:,:], xv[:,:,50:150,:,:], xv[:,:,150:300,:,:]
        ylist = [y1,y2,y3]
          
        # temporaily set the whole batch uses same permutation order
        # now changed to the every sample uses different permutation
        
        order_final = np.asarray(select_order_label[order_index[0]-1]).reshape((1,3))
        for order in order_index[1:]:   
            order_row = np.asarray(select_order_label[order-1]).reshape((1,3))
            order_final = np.concatenate((order_final,order_row),axis=0)
                  
        # firstly we adapt the batch data
        out_list = list()
        x_list = list()
        
        global y
        # define every GPU device
        if str(x.device)=='cuda:0':
            y = range(0,N)
        if str(x.device)=='cuda:1':
            y = range(N,2*N)
        if str(x.device)=='cuda:2':
            y = range(2*N,3*N)
        if str(x.device)=='cuda:3':
            y = range(3*N,4*N)
        
        for i in y:
            if random.randint(0,1) == 0:
                xx = xlist[order_final[i][0]-1][i%N,:,:,:,:]
            else:
                xx = ylist[order_final[i][0]-1][i%N,:,:,:,:]
            for j in range(1,3):
                if random.randint(0,1) == 0:
                    xi = xlist[order_final[i][j]-1][i%N,:,:,:,:]
                else:
                    xi = ylist[order_final[i][j]-1][i%N,:,:,:,:]
                xx = torch.cat((xx,xi),dim=1)
                
            x_list.append(xx)
        
        xj = x_list[0]
        xj = torch.unsqueeze(xj, dim=0)
        if N>1:
            for i in range(1,N):
                xj = torch.cat((xj,torch.unsqueeze(x_list[i], dim=0)),dim=0)
        
        # sequence segmanting and adding noise
        x1, x2, x3 = xj[:,:,0:50,:,:], xj[:,:,50:150,:,:], xj[:,:,150:300,:,:]
        xj_list = [x1,x2,x3]
               
        for xj in xj_list:
            xj = torch.cat((xj,torch.randn((N,C,5,V,M)).cuda()),dim=2)
            Nj, Cj, Tj, Vj, Mj = xj.size()
            xj = xj.permute(0, 4, 3, 1, 2).contiguous().view(Nj, Mj * Vj * Cj, Tj)
            xj = self.data_bn(xj)
            xj = xj.view(Nj * Mj, Vj, Cj, Tj).permute(0,2,3,1).contiguous()
        
            xj = F.relu(self.sgcn1(xj) + self.gcn3d1(xj), inplace=True)
            xj = self.tcn1(xj)
            xj = F.relu(self.sgcn2(xj) + self.gcn3d2(xj), inplace=True)
            xj = self.tcn2(xj)
            xj = F.relu(self.sgcn3(xj) + self.gcn3d3(xj), inplace=True)
            xj = self.tcn3(xj)

            outj = xj
            out_channels = outj.size(1)
            outj = outj.view(Nj, Mj, out_channels, -1)
            outj = outj.mean(3)   
            outj = outj.mean(1)   
            outj = self.fc3(outj)
            out_list.append(outj)
                        
        outj = torch.cat((out_list[0],out_list[1],out_list[2]),dim=1)
        outj = self.fc1(outj)
        outj = self.fc4(outj)
        '''

        '''
        #the same as above, firstly we adapt the batch data
        xl_list = list()
        choice_final = np.asarray(choice).reshape((-1,1))
                
        for i in y:
            if choice_final[i][0] == 1:
                xl = torch.cat((torch.zeros(1,C,T,8,M).cuda(),torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([8,9,10,11]).cuda()),torch.zeros(1,C,T,11,M).cuda(),torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([23,24]).cuda())), dim=3)
            if choice_final[i][0] == 2:
                xl = torch.cat((torch.zeros(1,C,T,4,M).cuda(),torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([4,5,6,7]).cuda()),torch.zeros(1,C,T,13,M).cuda(),torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([21,22]).cuda()), torch.zeros(1,C,T,2,M).cuda()), dim=3)
            if choice_final[i][0] == 3:
                xl = torch.cat((torch.zeros(1,C,T,16,M).cuda(),torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([16,17,18,19]).cuda()),torch.zeros(1,C,T,5,M).cuda()), dim=3)
            if choice_final[i][0] == 4:
                xl = torch.cat((torch.zeros(1,C,T,12,M).cuda(),torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([12,13,14,15]).cuda()),torch.zeros(1,C,T,9,M).cuda()), dim=3)
            if choice_final[i][0] == 5:
                xl = torch.cat((torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([0,1,2,3,4]).cuda()),torch.zeros(1,C,T,3,M).cuda(),torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([8]).cuda()), torch.zeros(1,C,T,3,M).cuda(), torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([12]).cuda()), torch.zeros(1,C,T,3,M).cuda(), torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([16]).cuda()), torch.zeros(1,C,T,3,M).cuda(), torch.index_select(x[i%2,:,:,:,:].reshape(1,C,T,V,M), 3, torch.tensor([20]).cuda()), torch.zeros(1,C,T,4,M).cuda()), dim=3)
            xl_list.append(xl)
        
        xl = xl_list[0]
        if N>1:
            for i in range(1,N):
                xl = torch.cat((xl,xl_list[i]),dim=0)  
        
        xl = xl.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        xl = self.data_bn(xl)
        xl = xl.view(N * M, V, C, T).permute(0,2,3,1).contiguous()     
        xl = F.relu(self.sgcn1(xl) + self.gcn3d1(xl), inplace=True)
        xl = self.tcn1(xl)
        xl = F.relu(self.sgcn2(xl) + self.gcn3d2(xl), inplace=True)
        xl = self.tcn2(xl)
        xl = F.relu(self.sgcn3(xl) + self.gcn3d3(xl), inplace=True)
        xl = self.tcn3(xl)
        
        outl = xl
        out_channels = outl.size(1)
        outl = outl.view(N, M, out_channels, -1)
        outl = outl.mean(3)   
        outl = outl.mean(1) 
        outl = self.fc3(outl)
        outl = self.fc2(outl)
        '''

        '''
        xp = x[:,:,0:50,:,:]
        p = random.randint(1,50)
        xp1 = torch.cat((xp[:,:,0:p,:,:],torch.randn((N,C,5,V,M)).cuda()),dim=2)
        if p==50:
            xp = xp1
        else:
            xp = torch.cat((xp1,xp[:,:,p:50,:,:]),dim=2)
        
        Np, Cp, Tp, Vp, Mp = xp.size()
        xp = xp.permute(0, 4, 3, 1, 2).contiguous().view(Np, Mp * Vp * Cp, Tp)
        xp = self.data_bn(xp)
        xp = xp.view(Np * Mp, Vp, Cp, Tp).permute(0,2,3,1).contiguous()
        xp = F.relu(self.sgcn1(xp) + self.gcn3d1(xp), inplace=True)
        xp = self.tcn1(xp)
        xp = F.relu(self.sgcn2(xp) + self.gcn3d2(xp), inplace=True)
        xp = self.tcn2(xp)
        xp = F.relu(self.sgcn3(xp) + self.gcn3d3(xp), inplace=True)
        xp = self.tcn3(xp)       
        xp = self.gcnr(xp)
        xp = self.gcnr1(xp)
        xp = self.tcnr1(xp)
        xp = self.tcnr(xp)
        outp = xp
        
        outp_gt = xv[:,:,50:200,:,:]
        outp_gt = outp_gt.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, 150)
        outp_gt = self.data_bn(outp_gt)
        outp_gt = outp_gt.view(N * M, V, C, 150).permute(0,2,3,1).contiguous()
        '''
        
        '''
        xr = x[:,:,0:150,:,:]
        xr = xr.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, 150)
        xr = self.data_bn(xr)
        xr = xr.view(N * M, V, C, 150).permute(0,2,3,1).contiguous()
        xr = F.relu(self.sgcn1(xr) + self.gcn3d1(xr), inplace=True)
        xr = self.tcn1(xr)
        xr = F.relu(self.sgcn2(xr) + self.gcn3d2(xr), inplace=True)
        xr = self.tcn2(xr)
        xr = F.relu(self.sgcn3(xr) + self.gcn3d3(xr), inplace=True)
        xr = self.tcn3(xr)
        print(xr.shape)
        xr = self.reconstructions(xr)
        outr = xr
        outr_gt = xv[:,:,150:300,:,:]
        '''

        N, C, T, V, M = x.size()
        choice = random.randint(1,5)
        if choice == 1:
            xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4,5,6,7,8]).cuda()), torch.zeros(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([12,13,14,15,16,17,18,19,20,21,22]).cuda()), torch.zeros(N,C,T,2,M).cuda()), dim=3)
        if choice == 2:
            xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4]).cuda()), torch.zeros(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([8,9,10,11,12,13,14,15,16,17,18,19,20]).cuda()), torch.zeros(N,C,T,2,M).cuda(), torch.index_select(x, 3, torch.tensor([23,24]).cuda())), dim=3)
        if choice == 3:
            xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).cuda()), torch.zeros(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([20,21,22,23,24]).cuda())), dim=3)
        if choice == 4:
            xj = torch.cat((torch.index_select(x, 3, torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12]).cuda()), torch.zeros(N,C,T,3,M).cuda(), torch.index_select(x, 3, torch.tensor([16,17,18,19,20,21,22,23,24]).cuda())), dim=3)
        if choice == 5:
            xj = torch.cat((torch.zeros(N,C,T,4,M).cuda(), torch.index_select(x, 3, torch.tensor([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).cuda()), torch.zeros(N,C,T,1,M).cuda(), torch.index_select(x, 3, torch.tensor([21,22,23,24]).cuda())), dim=3)
        
        '''
        Nj, Cj, Tj, Vj, Mj = xj.size()
        xj = xj.permute(0, 4, 3, 1, 2).contiguous().view(Nj, Mj * Vj * Cj, Tj)
        if (choice == 1)|(choice == 2)|(choice == 5):
            xj = self.data_bn1(xj)
        if (choice == 3)|(choice == 4):
            xj = self.data_bn2(xj)
        xj = xj.view(Nj * Mj, Vj, Cj, Tj).permute(0,2,3,1).contiguous()
        '''

        xj = torch.cat((xj[:,:,:200,:,:], torch.zeros(N,C,300,V,M).cuda()),dim=2)
        
        # for every choice, define respective forward process
        '''
        if choice == 5:
            xj = F.relu(self.sgcn15(xj) + self.gcn3d15(xj), inplace=True)
            xj = self.tcn1(xj)
            xj = F.relu(self.sgcn25(xj) + self.gcn3d25(xj), inplace=True)
            xj = self.tcn2(xj)
            xj = F.relu(self.sgcn35(xj) + self.gcn3d35(xj), inplace=True)
            xj = self.tcn3(xj)
        
        if choice == 1:
            xj = F.relu(self.sgcn11(xj) + self.gcn3d11(xj), inplace=True)
            xj = self.tcn1(xj)
            xj = F.relu(self.sgcn21(xj) + self.gcn3d21(xj), inplace=True)
            xj = self.tcn2(xj)
            xj = F.relu(self.sgcn31(xj) + self.gcn3d31(xj), inplace=True)
            xj = self.tcn3(xj)
            
        if choice == 2:
            xj = F.relu(self.sgcn12(xj) + self.gcn3d12(xj), inplace=True)
            xj = self.tcn1(xj)
            xj = F.relu(self.sgcn22(xj) + self.gcn3d22(xj), inplace=True)
            xj = self.tcn2(xj)
            xj = F.relu(self.sgcn32(xj) + self.gcn3d32(xj), inplace=True)
            xj = self.tcn3(xj)
            
        if choice == 3:
            xj = F.relu(self.sgcn13(xj) + self.gcn3d13(xj), inplace=True)
            xj = self.tcn1(xj)
            xj = F.relu(self.sgcn23(xj) + self.gcn3d23(xj), inplace=True)
            xj = self.tcn2(xj)
            xj = F.relu(self.sgcn33(xj) + self.gcn3d33(xj), inplace=True)
            xj = self.tcn3(xj)
            
        if choice == 4:
            xj = F.relu(self.sgcn14(xj) + self.gcn3d14(xj), inplace=True)
            xj = self.tcn1(xj)
            xj = F.relu(self.sgcn24(xj) + self.gcn3d24(xj), inplace=True)
            xj = self.tcn2(xj)
            xj = F.relu(self.sgcn34(xj) + self.gcn3d34(xj), inplace=True)
            xj = self.tcn3(xj)
        '''

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)
        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)
        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)
        
        xj = xj.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        xj = self.data_bn(xj)
        xj = xj.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        xj = xj.half()

        xj = F.relu(self.sgcn1(xj) + self.gcn3d1(xj), inplace=True)
        xj = self.tcn1(xj)
        xj = F.relu(self.sgcn2(xj) + self.gcn3d2(xj), inplace=True)
        xj = self.tcn2(xj)
        xj = F.relu(self.sgcn3(xj) + self.gcn3d3(xj), inplace=True)
        xj = self.tcn3(xj)

        out1 = xj
        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)
        outtemp = out.mean(1)
        out = self.fc(outtemp)

        xm = xm.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        xm = self.data_bn(xm)
        xm = xm.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        xm = F.relu(self.sgcn1(xm) + self.gcn3d1(xm), inplace=True)
        xm = self.tcn1(xm)
        xm = F.relu(self.sgcn2(xm) + self.gcn3d2(xm), inplace=True)
        xm = self.tcn2(xm)
        xm = F.relu(self.sgcn3(xm) + self.gcn3d3(xm), inplace=True)
        xm = self.tcn3(xm)

        outm = xm
        out_channelsm = outm.size(1)
        outm = outm.view(N, M, out_channelsm, -1)
        outm = outm.mean(3)
        outm = outm.mean(1)
        out2 = self.prototype(outm)

        return out1, out, out2
        
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
        outv = self.mlp1(outv)
        outv = self.bn1(outv)
        outv = self.relu1(outv)
        outv = self.mlp2(outv)
        outv = self.bn2(outv)
        outv = self.relu2(outv)
        outv = self.mlp3(outv)
        '''

        return
        
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
        
        return x, trans
            

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
        '''
        self.encoderv_t = Model(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 label=0)
        
        self.encoderv_s = Model(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 label=0)
        
        self.encoderp_t = Model(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 label=1)
        '''     
        self.encoderp_s = Model(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3)
        
        self.decoder = Model1(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3)
        '''         
        self.transp_t = Model2(num_class,
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

        self.data_bn = nn.BatchNorm1d(150)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)
               
    def forward(self, x, xm, viewt, t):

        viewt = list(viewt)
        t = list(t)
        #view1 = list()
        #view2 = list()
        #xv, viewt1 = _transform_rotation(x)
        x = _transform_shape(x)
        xm = _transform_shape(xm)
        N, C, T, V, M = x.size()
        
        '''
        if random.randint(0,1)==0:
            x1 = x[0,:,:,:,:].view(1,C,T,V,M)
            view1.append(viewt[0])
            x2 = xv[0,:,:,:,:].view(1,C,T,V,M)
            view2.append(8+(viewt1[0]+1)*(viewt[0]+1)-1)
        else:
            x1 = xv[0,:,:,:,:].view(1,C,T,V,M)
            view1.append(8+(viewt1[0]+1)*(viewt[0]+1)-1)
            x2 = x[0,:,:,:,:].view(1,C,T,V,M)
            view2.append(viewt[0])
               
        for i in range(1,N):
            if random.randint(0,1)==0:
                x1 = torch.cat([x1,x[i,:,:,:,:].view(1,C,T,V,M)],dim=0)
                view1.append(viewt[i])
                x2 = torch.cat([x2,xv[i,:,:,:,:].view(1,C,T,V,M)],dim=0)
                view2.append(8+(viewt1[i]+1)*(viewt[i]+1)-1)
            else:
                x1 = torch.cat([x1,xv[i,:,:,:,:].view(1,C,T,V,M)],dim=0)
                view1.append(8+(viewt1[i]+1)*(viewt[i]+1)-1)
                x2 = torch.cat([x2,x[i,:,:,:,:].view(1,C,T,V,M)],dim=0)
                view2.append(viewt[i])
        '''

        # true+virtual channel: x1; true channel: x2
        x1 = torch.split(x, [int(N/2),int(N/2)], dim=0)[0]
        x2 = torch.split(x, [int(N/2),int(N/2)], dim=0)[1]
        xm1 = torch.split(xm, [int(N/2),int(N/2)], dim=0)[0]
        xm2 = torch.split(xm, [int(N/2),int(N/2)], dim=0)[1]

        '''
        for i in range(N):
             view1.append(2*viewt[i])
             view2.append(2*viewt[i])
             if viewt1[i]%2==0:
                 view1.append(2*viewt[i]+viewt1[i]+1)
             else:
                 view1.append(viewt[i]+viewt1[i]+1)
        '''

        out1_gt = x1
        out2_gt = x2
        out1_gt = out1_gt.permute(0, 4, 3, 1, 2).contiguous().view(int(N/2), M * V * C, T)
        out1_gt = self.data_bn(out1_gt)
        out1_gt = out1_gt.view(int(N/2) * M, V, C, T).permute(0,2,3,1).contiguous()
        out2_gt = out2_gt.permute(0, 4, 3, 1, 2).contiguous().view(int(N/2), M * V * C, T)
        out2_gt = self.data_bn(out2_gt)
        out2_gt = out2_gt.view(int(N/2) * M, V, C, T).permute(0,2,3,1).contiguous()
        out_gt = torch.cat((out1_gt,out2_gt), dim=0)

        out1vr, out1_v, mm = self.encoderp_s(x1,xm1)
        x1, trans1 = self.transp_s(x1)
        out1pr, out1_p, out1_cp11 = self.encoderp_s(x1,xm1)
        xm1, mm = self.transp_s(xm1)
        out1pr, out1_p, out1_cp12 = self.encoderp_s(x1,xm1)
        #out1_cp = out1_cp[0:N,:]

        out1vr = torch.abs(out1vr-out1pr)
        out1_v = torch.abs(out1_v-out1_p)
        #out1vr = torch.div(out1vr,out1pr)
        #out1_v = torch.div(out1_v,out1_p)
        out1_vn = out1_v
        out1_vn = self.fc1(out1_vn)
        out1_vn = F.relu(out1_vn)
        out1_vn = self.fc2(out1_vn)
        out1_vn = F.relu(out1_vn)
        out1_vn = self.fc3(out1_vn)
                
        out2vr, out2_v, mm = self.encoderp_s(x2,xm2)
        x2, trans2 = self.transp_s(x2)
        out2pr, out2_p, out2_cp21 = self.encoderp_s(x2,xm2)
        xm2, mm = self.transp_s(xm2)
        out2pr, out2_p, out2_cp22 = self.encoderp_s(x2,xm2)
        out2vr = torch.abs(out2vr-out2pr)
        out2_v = torch.abs(out2_v-out2_p)
        #out2vr = torch.div(out2vr,out2pr)
        #out2_v = torch.div(out2_v,out2_p)
        out2_vn = out2_v
        out2_vn = self.fc1(out2_vn)
        out2_vn = F.relu(out2_vn)
        out2_vn = self.fc2(out2_vn)
        out2_vn = F.relu(out2_vn)
        out2_vn = self.fc3(out2_vn)
        
        out_vn = torch.cat((out1_vn,out2_vn),dim=0)
        trans = torch.cat((trans1,trans2),dim=0)
        
        outcp1 = torch.cat((out1_cp11, out2_cp21), dim=0)
        outcp2 = torch.cat((out1_cp12, out2_cp22), dim=0)
        outcp = torch.stack((outcp1, outcp2), dim=1)
        
        out1r = torch.add(out1vr,out2pr)
        out2r = torch.add(out2vr,out1pr)
        out1 = self.decoder(out1r)
        out2 = self.decoder(out2r)
        out = torch.cat((out1, out2), dim=0)
        
        #feature1 = torch.mm(out1_v, out1_v.T)
        #feature2 = torch.mm(out2_v, out2_v.T)
        #amp.register_float_function(torch, 'sigmoid')
        #feature1 = torch.sigmoid(feature1)
        #feature2 = torch.sigmoid(feature2)
        #out1_pf = torch.cat((out2_p,out1_p),dim=0)
        #out1_vf = torch.cat((out2_v,out1_v),dim=0)
        #outfinal_v = torch.cat((out1_v,out2_v),dim=0)
        #outfinal_p = torch.cat((out1_p,out2_p),dim=0)
        #out = self.encoder(torch.split(x,int(N/2),dim=0)[0])
        #outv = self.encoder(torch.split(x,int(N/2),dim=0)[1])

        '''
        if random.randint(0,1)==0:
            x1 = out[0,:].view(1,-1)
            view1.append(viewt[0])
            x2 = outv[0,:].view(1,-1)
            view2.append(viewt[0]+view[0])
        else:
            x1 = outv[0,:].view(1,-1)
            view1.append(viewt[0]+view[0])
            x2 = out[0,:].view(1,-1)
            view2.append(viewt[0])
                
        for i in range(1,N):
            if random.randint(0,1)==0:
                x1 = torch.cat([x1,out[i,:].view(1,-1)],dim=0)
                view1.append(viewt[i])
                x2 = torch.cat([x2,outv[i,:].view(1,-1)],dim=0)
                view2.append(viewt[i]+view[i])
            else:
                x1 = torch.cat([x1,outv[i,:].view(1,-1)],dim=0)
                view1.append(viewt[i]+view[i])
                x2 = torch.cat([x2,out[i,:].view(1,-1)],dim=0)
                view2.append(viewt[i])               
        '''

        #diffp = torch.zeros(1).cuda()
        #diffv = torch.zeros(1).cuda()
        #count = 0
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
        feature = torch.mm(outfinal, outfinal.T)
        label = torch.zeros(2*N,2*N).cuda()
        for i in range(2*N):
            for j in range(2*N):
                if i != j:
                    if t[i%N] == t[j%N]:
                        label[i,j]=1

        #index = torch.tensor([[1,2,3,4,5,6,7], [0,2,3,4,5,6,7], [0,1,3,4,5,6,7], [0,1,2,4,5,6,7], [0,1,2,3,5,6,7], [0,1,2,3,4,6,7], [0,1,2,3,4,5,7], [0,1,2,3,4,5,6]]).cuda()
        index = torch.tensor([[1,2,3],[0,2,3],[0,1,3],[0,1,2]]).cuda()
        diff = feature.gather(1,index)
        label = label.gather(1,index)      
        '''

        #c = self.bnlast(x1).T @ self.bnlast(x2)
        #c.div_(N)
        #on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(1/32)
        #off_diag = off_diagonal(c).pow_(2).sum().mul(1/32)
        #loss1 = (diff1 + diff2)
        #out1 = self.encoder_t(torch.split(x,int(N/2),dim=0)[0])
        #outv1 = self.encoder_s(torch.split(x,int(N/2),dim=0)[1])

        '''
        if random.randint(0,1)==0:
            x11 = out1[0,:].view(1,-1)
            x21 = outv1[0,:].view(1,-1)
        else:
            x11 = outv1[0,:].view(1,-1)
            x21 = out1[0,:].view(1,-1)
                
        for i in range(1,N):
            if random.randint(0,1)==0:
                x11 = torch.cat([x11,out1[i,:].view(1,-1)],dim=0)
                x21 = torch.cat([x21,outv1[i,:].view(1,-1)],dim=0)
            else:
                x11 = torch.cat([x11,outv1[i,:].view(1,-1)],dim=0)
                x21 = torch.cat([x21,out1[i,:].view(1,-1)],dim=0)               
        '''

        '''
        diff2 = torch.zeros(1).cuda()
        for index in range(int(N/2)):
            for index in range(int(N/2)):
                sim1 = torch.cosine_similarity(out1[index,:],outv1[index:],dim=0).cuda()
                diff2 = diff2 + sim1.add_(-1).pow_(2)
                
                if view1[index]==view2[index]:
                    diff11 = diff11 + sim1.add_(-1).pow_(2)                
                else:
                    diff21 = diff21 + sim1.pow_(2)                
                #count1 = count1 + 1
       
        #loss2 = (diff11 + diff21)
        diff2 = diff2/4
        '''

        #on_diag = torch.diagonal(feature1).add_(-1).pow_(2).sum() + torch.diagonal(feature2).add_(-1).pow_(2).sum()
        #off_diag = off_diagonal(feature1).pow_(2).sum() + off_diagonal(feature2).pow_(2).sum()
        #regular = on_diag + off_diag
        #regular = regular/(2*N*N)
        #diff = (diffp+diffv)/64

        return out, out_gt, diff, label, outcp, out_vn, trans

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

        #self.net = torch.nn.DataParallel(self.net)
        #self.net = torch.nn.parallel.DistributedDataParallel(self.net)
               
    def forward(self, x, xm, viewt, t):
        
        return self.net(x, xm, viewt, t)
    
'''    
class MyMsg3d4jigsaw(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(MyMsg3d4jigsaw, self).__init__()
      
        self.encoder1 = Model4jigsaw(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3)

        self.encoder1 = torch.nn.DataParallel(self.encoder1)
        
        for m in self.modules():
            self.weights_init(m)
               
    def forward(self, x):
    
        x_v = _transform_rotation(x)
        #x_v = torch.tensor(x_v, requires_grad=True)
        x= _transform_shape(x)
        
        return self.encoder1(x)
    
    def weights_init(self, m):
        
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

class MyMsg3d4location(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(MyMsg3d4location, self).__init__()

        self.encoder2 = Model4location(num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3)
                 
        self.encoder2 = torch.nn.DataParallel(self.encoder2)
        
        for m in self.modules():
            self.weights_init(m)
               
    def forward(self, x):
    
        x_v = _transform_rotation(x)
        #x_v = torch.tensor(x_v, requires_grad=True)
        x= _transform_shape(x)
        
        return self.encoder2(x)
    
    def weights_init(self, m):
        
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
'''

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

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
            #x[i][j] = random.choice([0,math.pi/36,math.pi/18,3*math.pi/36,math.pi/9,5*math.pi/36,6*math.pi/36,7*math.pi/36,8*math.pi/36,math.pi/4])
            x[i][j] = 0
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
            #z[i][j] = random.choice([0,math.pi/36,math.pi/18,3*math.pi/36,math.pi/9,5*math.pi/36,6*math.pi/36,7*math.pi/36,8*math.pi/36,math.pi/4])
            z[i][j] = 0
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
    inputs1 = inputs1.contiguous().view(-1, 3, 3014, 25)
    inputs1 = torch.split(inputs1, [500,2514], dim=2)[0]

    b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55 = a2.chunk(25, dim=2)
    inputs2 = torch.stack([b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55], dim=3)
    inputs2 = inputs2.permute(0, 2, 1, 3)
    inputs2 = inputs2.contiguous().view(-1, inputs2.size()[1], inputs2.size()[2] * inputs2.size()[3])
    inputs2 = torch.cat((inputs2, Variable(inputs2.data.new(inputs2.size()[0], 1, inputs2.size()[2]).fill_(1))), dim=1).cuda()
    inputs2 = torch.matmul(trans.half(), inputs2)
    inputs2 = torch.matmul(rot.half(), inputs2)
    inputs2 = inputs2.contiguous().view(-1, 3, 3014, 25)
    inputs2 = torch.split(inputs2, [500,2514], dim=2)[0]

    inputs = torch.stack([inputs1, inputs2], dim=4)

    return inputs,view

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
    
def hmd(x, y):
    count = 0
    for idx, m in enumerate(list(x)):
        for idx1, n in enumerate(list(y)):
            if idx == idx1:
                 z = m ^ n
                 while z != 0:
                     if z & 1 == 1:
                         count += 1
                     z = z >> 1
    return count
    
def proto_contrast(logit, prob):
    b, n, k = logit.shape
    q_s = sinkhorn(prob[:, 0])[-b:]
    q_t = sinkhorn(prob[:, 1])[-b:]
    #pq_s = q_s[-b*(ngpus-replica_id): -b*(ngpus-replica_id-1)] if (ngpus-replica_id-1) > 0 else q_s[-b*(ngpus-replica_id):]
    #pq_t = q_t[-b*(ngpus-replica_id): -b*(ngpus-replica_id-1)] if (ngpus-replica_id-1) > 0 else q_t[-b*(ngpus-replica_id):]
    prob = F.softmax(logit/0.1, dim=-1)
    loss = - 0.5 * (torch.sum(q_t*torch.log(prob[:, 0]+1e-10), dim=-1)+
                    torch.sum(q_s*torch.log(prob[:, 1]+1e-10), dim=-1))
    return loss
    
def sinkhorn(scores, eps=0.05, niters=3):
    with torch.no_grad():
        M = torch.max(scores/eps)
        Q = scores/eps - M
        Q = torch.exp(Q).transpose(0, 1)
        Q = shoot_infs(Q)
        Q = Q / torch.sum(Q)
        K, B = Q.shape
        u, r, c = torch.zeros(K), torch.ones(K)/K, torch.ones(B)/B
        for _ in range(niters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q = Q * u.unsqueeze(1)
            Q = Q * (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).transpose(0, 1)

def shoot_infs(inp_tensor):
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


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

    N, C, T, V, M = 8, 3, 500, 25, 2
    x = torch.randn(N,C,T,V,M)
    model.forward(x)

    print('Model total # params:', count_params(model))
