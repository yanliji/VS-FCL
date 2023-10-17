import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import tqdm
import random
from itertools import permutations
from .temporal_transformer_windowed import tcn_unit_attention_block
from .temporal_transformer import tcn_unit_attention
from .gcn_attention import gcn_unit_attention
from .net import Unit2D, conv_init, import_class
from .unit_gcn import unit_gcn
from .unit_agcn import unit_agcn
from graph.ntu_rgb_d import Graph


default_backbone_all_layers = [(3, 64, 1), (64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),(128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]
default_backbone = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]

class Model(nn.Module):

    def __init__(self,
                 channel,
                 num_class,
                 window_size,
                 num_point,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 attention_3,
                 relative,
                 kernel_temporal,
                 double_channel,
                 drop_connect,
                 concat_original,
                 dv,
                 dk,
                 Nh,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 all_layers,
                 data_normalization,
                 visualization,
                 skip_conn,
                 adjacency,
                 bn_flag,
                 weight_matrix,
                 device,
                 n,
                 more_channels,
                 num_person=1,
                 use_data_bn=False,
                 backbone_config=None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5,
                 agcn = True):
        super(Model, self).__init__()

        self.graph = Graph()
        self.A = torch.from_numpy(self.graph.A.astype(np.float32))

        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.multiscale = multiscale
        self.attention = attention
        self.tcn_attention = tcn_attention
        self.drop_connect = drop_connect
        self.more_channels = more_channels
        self.concat_original = concat_original
        self.all_layers = all_layers
        self.dv = dv
        self.num = n
        self.Nh = Nh
        self.dk = dk
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.visualization = visualization
        self.double_channel = double_channel
        self.adjacency = adjacency
        self.M_dim_bn = True
        if self.M_dim_bn:
            self.data_bn = nn.BatchNorm1d(channel * num_point * num_person)
        else:
            self.data_bn = nn.BatchNorm1d(channel * num_point)

        if self.all_layers:
            if not self.double_channel:
                self.starting_ch = 64
            else:
                self.starting_ch = 128
        else:
            if not self.double_channel:
                self.starting_ch = 128
            else:
                self.starting_ch = 256

        kwargs = dict(
            A=self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn,
            dropout=dropout,
            kernel_size=temporal_kernel_size,
            attention=attention,
            only_attention=only_attention,
            tcn_attention=tcn_attention,
            only_temporal_attention=only_temporal_attention,
            attention_3=attention_3,
            relative=relative,
            weight_matrix=weight_matrix,
            device=device,
            more_channels=self.more_channels,
            drop_connect=self.drop_connect,
            data_normalization=self.data_normalization,
            skip_conn=self.skip_conn,
            adjacency=self.adjacency,
            starting_ch=self.starting_ch,
            visualization=self.visualization,
            all_layers=self.all_layers,
            dv=self.dv,
            dk=self.dk,
            Nh=self.Nh,
            num=n,
            dim_block1=dim_block1,
            dim_block2=dim_block2,
            dim_block3=dim_block3,
            num_point=num_point,
            agcn = agcn
        )

        if self.multiscale:
            unit = TCN_GCN_unit_multiscale
        else:
            unit = TCN_GCN_unit

        if backbone_config is None:
            if self.all_layers:
                backbone_config = default_backbone_all_layers
            else:
                backbone_config = default_backbone
        self.backbone = nn.ModuleList([
            unit(in_c, out_c, stride=stride, **kwargs)
            for in_c, out_c, stride in backbone_config
        ])
        if self.double_channel:
            backbone_in_c = backbone_config[0][0] * 2
            backbone_out_c = backbone_config[-1][1] * 2
        else:
            backbone_in_c = backbone_config[0][0]
            backbone_out_c = backbone_config[-1][1]
        backbone_out_t = window_size
        backbone = []
        for i, (in_c, out_c, stride) in enumerate(backbone_config):
            if self.double_channel:
                in_c = in_c * 2
                out_c = out_c * 2
            if i == 3 and concat_original:
                backbone.append(unit(in_c + channel, out_c, stride=stride, last=i == len(default_backbone) - 1, last_graph=(i == len(default_backbone) - 1), layer=i, **kwargs))
            else:
                backbone.append(unit(in_c, out_c, stride=stride, last=i == len(default_backbone) - 1, last_graph=(i == len(default_backbone) - 1), layer=i, **kwargs))
            if backbone_out_t % stride == 0:
                backbone_out_t = backbone_out_t // stride
            else:
                backbone_out_t = backbone_out_t // stride + 1
        self.backbone = nn.ModuleList(backbone)
        for i in range(0, len(backbone)):
            pytorch_total_params = sum(p.numel() for p in self.backbone[i].parameters() if p.requires_grad)

        '''
        self.gcnr1 = unit_gcn(
                    channel,
                    64,
                    self.A,
                    mask_learning=mask_learning,
                    use_local_bn=use_local_bn)
        self.gcnr2 = unit_gcn(
                    64,
                    128,
                    self.A,
                    mask_learning=mask_learning,
                    use_local_bn=use_local_bn)
        self.fcnr = nn.Conv1d(128, 6, kernel_size=1)    
        '''

        if not all_layers:
            if not agcn:
                self.gcn0 = unit_gcn(
                    channel,
                    backbone_in_c,
                    self.A,
                    mask_learning=mask_learning,
                    use_local_bn=use_local_bn)
            else:
                self.gcn0 = unit_agcn(
                    channel,
                    backbone_in_c,
                    self.A,
                    mask_learning=mask_learning,
                    use_local_bn=use_local_bn)

            self.tcn0 = Unit2D(backbone_in_c, backbone_in_c, kernel_size=9)

        self.person_bn = nn.BatchNorm1d(backbone_out_c)
        self.gap_size = backbone_out_t
        self.fcn = nn.Conv1d(backbone_out_c, num_class, kernel_size=1)
        #self.fcn1 = nn.Conv1d(backbone_out_c, 128, kernel_size=1)
        #self.fcn2 = nn.Conv1d(384, 6, kernel_size=1)
        conv_init(self.fcn)
        self.mlp1 = nn.Linear(backbone_out_c, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.mlp2 = nn.Linear(256, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.mlp3 = nn.Linear(128, 80, bias=False)

    def forward(self, x):
        # define lists and dictionaries
        #order_list = list(permutations([1, 2, 3, 4, 5], 5))
        #order_list = list(permutations([1, 2, 3], 3))
        #order_label = dict()
        #select_order_label = dict()
        #count = dict()
        # define the whole permutation sets
        #for idx, element in enumerate(order_list):
        #    order_label[idx] = order_list[idx]
        #select_order_label = order_label
        #x1, x2, x3 = x[:,:,0:50,:,:], x[:,:,50:150,:,:], x[:,:,150:300,:,:]
        #xlist = [x1,x2,x3]
        #y1, y2, y3 = xv[:,:,0:50,:,:], xv[:,:,50:150,:,:], xv[:,:,150:300,:,:]
        #ylist = [y1,y2,y3]
        #order_final = np.asarray(select_order_label[order_index[0]-1]).reshape((1,3))
        #for order in order_index[1:]:   
        #    order_row = np.asarray(select_order_label[order-1]).reshape((1,3))
        #    order_final = np.concatenate((order_final,order_row),axis=0)
        # firstly we adapt the batch data
        #out_list = list()
        #x_list = list()
        #global y
        # define every GPU device
        #if str(x.device)=='cuda:0':
        #    y = range(0,N)
        #if str(x.device)=='cuda:1':
        #    y = range(N,2*N)
        #if str(x.device)=='cuda:2':
        #    y = range(2*N,3*N)
        #if str(x.device)=='cuda:3':
        #    y = range(3*N,4*N)
        #for i in y:
        #    if random.randint(0,1) == 0:
        #        xx = xlist[order_final[i][0]-1][i%N,:,:,:,:]
        #    else:
        #        xx = ylist[order_final[i][0]-1][i%N,:,:,:,:]
        #    for j in range(1,3):
        #        if random.randint(0,1) == 0:
        #            xi = xlist[order_final[i][j]-1][i%N,:,:,:,:]
        #        else:
        #            xi = ylist[order_final[i][j]-1][i%N,:,:,:,:]
        #        xx = torch.cat((xx,xi),dim=1)
        #    x_list.append(xx)
        #xj = x_list[0]
        #xj = torch.unsqueeze(xj, dim=0)
        #if N>1:
        #    for i in range(1,N):
        #        xj = torch.cat((xj,torch.unsqueeze(x_list[i], dim=0)),dim=0)
        # sequence segmanting and adding noise
        #x1, x2, x3 = xj[:,:,0:50,:,:], xj[:,:,50:150,:,:], xj[:,:,150:300,:,:]
        #j_list = [x1,x2,x3]
        #for xj in xj_list:
        #    Nj, Cj, Tj, Vj, Mj = xj.size()
        #    xj = torch.cat((xj,torch.randn((N,C,5,V,M)).cuda(),torch.randn((N,C,295-Tj,V,M)).cuda()),dim=2)
        #    if (self.concat_original):
        #        xj_coord = xj
        #        xj_coord = xj_coord.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)
        #    if self.use_data_bn:
        #        if self.M_dim_bn:
        #            xj = xj.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        #        else:
        #            xj = xj.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        #        xj = self.data_bn(xj)
        #        xj = xj.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        #    else:
        #        xj = xj.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        #    if not self.all_layers:
        #        xj = self.gcn0(xj)
        #        xj = self.tcn0(xj)
        #    for i, m in enumerate(self.backbone):
        #        if i == 3 and self.concat_original:
        #            xj = m(torch.cat((xj, xj_coord), dim=1))
        #        else:
        #            xj = m(xj)
        #   xj = F.avg_pool2d(xj, kernel_size=(1, V))
        #   c = xj.size(1)
        #    t = xj.size(2)
        #    xj = xj.view(N, M, c, t).mean(dim=1).view(N, c, t)
        #    xj = F.avg_pool1d(xj, kernel_size=xj.size()[2])
        #    xj = self.fcn1(xj)          
        #    out_list.append(xj)
        #outj = torch.cat((out_list[0],out_list[1],out_list[2]),dim=1)
        #outj = self.fcn2(outj)
        #outj = F.avg_pool1d(outj, outj.size()[2:])
        #outj = outj.view(N,6)
        #temp = x

        N, C, T, V, M = x.size()
        
        if (self.concat_original):
            x_coord = x
            x_coord = x_coord.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)

        if self.use_data_bn:
            if self.M_dim_bn:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        else:
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        '''
        x1 = self.gcnr1(x)
        x1 = self.gcnr2(x1)
        x1 = F.avg_pool2d(x1, x1.size()[2:])
        x1 = x1.view(N, M, -1, 1, 1).mean(dim=1)
        x1 = x1.view(N,128,1)
        x1 = self.fcnr(x1)
        trans = x1.view(x1.size(0), -1)       
        xv = _transform_s(temp,trans)
        '''

        if not self.all_layers:
            x = self.gcn0(x)
            x = self.tcn0(x)

        for i, m in enumerate(self.backbone):
            if i == 3 and self.concat_original:
                x = m(torch.cat((x, x_coord), dim=1))
            else:
                x = m(x)

        x = F.avg_pool2d(x, kernel_size=(1, V))
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)
        x = F.avg_pool1d(x, kernel_size=x.size()[2])

        x = x.view(x.size(0), -1)
        x = self.mlp1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.mlp2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.mlp3(x)

        '''
        if (self.concat_original):
            xv_coord = xv
            xv_coord = xv_coord.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)

        if self.use_data_bn:
            if self.M_dim_bn:
                xv = xv.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                xv = xv.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
            xv = self.data_bn(xv)
            xv = xv.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        else:
            xv = xv.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        if not self.all_layers:
            xv = self.gcn0(xv)
            xv = self.tcn0(xv)

        for i, m in enumerate(self.backbone):
            if i == 3 and self.concat_original:
                xv = m(torch.cat((xv, xv_coord), dim=1))
            else:
                xv = m(xv)

        xv = F.avg_pool2d(xv, kernel_size=(1, V))
        c = xv.size(1)
        t = xv.size(2)
        xv = xv.view(N, M, c, t).mean(dim=1).view(N, c, t)
        xv = F.avg_pool1d(xv, kernel_size=xv.size()[2])

        #xv = self.fcn1(xv)
        #xv = F.avg_pool1d(xv, xv.size()[2:])
        #xv = xv.view(N, 128)
        xv = xv.view(xv.size(0), -1)
        xv = self.mlp1(xv)
        xv = self.bn1(xv)
        xv = self.relu1(xv)
        xv = self.mlp2(xv)
        xv = self.bn2(xv)
        xv = self.relu2(xv)
        xv = self.mlp3(xv)
        '''

        return x
        
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class STTransformer(nn.Module):
    def __init__(self,
                 channel,
                 num_class,
                 window_size,
                 num_point,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 attention_3,
                 relative,
                 kernel_temporal,
                 double_channel,
                 drop_connect,
                 concat_original,
                 dv,
                 dk,
                 Nh,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 all_layers,
                 data_normalization,
                 visualization,
                 skip_conn,
                 adjacency,
                 bn_flag,
                 weight_matrix,
                 device,
                 n,
                 more_channels,
                 num_person=1,
                 use_data_bn=False,
                 backbone_config=None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5,
                 agcn = True):
        super(STTransformer, self).__init__()
        self.encoder = Model(channel,
                 num_class,
                 window_size,
                 num_point,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 attention_3,
                 relative,
                 kernel_temporal,
                 double_channel,
                 drop_connect,
                 concat_original,
                 dv,
                 dk,
                 Nh,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 all_layers,
                 data_normalization,
                 visualization,
                 skip_conn,
                 adjacency,
                 bn_flag,
                 weight_matrix,
                 device,
                 n,
                 more_channels,
                 num_person=1,
                 use_data_bn=False,
                 backbone_config=None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5,
                 agcn = True)
        '''
        self.encoder_t = Model(channel,
                 num_class,
                 window_size,
                 num_point,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 attention_3,
                 relative,
                 kernel_temporal,
                 double_channel,
                 drop_connect,
                 concat_original,
                 dv,
                 dk,
                 Nh,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 all_layers,
                 data_normalization,
                 visualization,
                 skip_conn,
                 adjacency,
                 bn_flag,
                 weight_matrix,
                 device,
                 n,
                 more_channels,
                 num_person=1,
                 use_data_bn=False,
                 backbone_config=None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5,
                 agcn = True)
        '''

        self.bnlast = nn.BatchNorm1d(80, affine=False)
        #self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        #self.encoder_k = torch.nn.DataParallel(self.encoder)
        
    
    def forward(self, xi):
        
        x = _transform_shape(xi)
        xv = _transform_rotation(xi)
        x = self.encoder(x)
        xv = self.encoder(xv)
        N = x.size()[0]
        
        if random.randint(0,1)==0:
            x1 = x[0,:].view(1,-1)
            x2 = xv[0,:].view(1,-1)
        else:
            x1 = xv[0,:].view(1,-1)
            x2 = x[0,:].view(1,-1)

        for i in range(1,N):
            if random.randint(0,1)==0:
                x1 = torch.cat([x1,x[i,:].view(1,-1)],dim=0)
                x2 = torch.cat([x2,xv[i,:].view(1,-1)],dim=0)
            else:
                x1 = torch.cat([x1,xv[i,:].view(1,-1)],dim=0)
                x2 = torch.cat([x2,x[i,:].view(1,-1)],dim=0)
            
        '''
        c = torch.tensor(np.zeros((60,60))).cuda()
        for i in range(N):
            c += x[i,:].reshape(1,-1).T @ xv[i,:].reshape(1,-1)
        '''

        c = self.bnlast(x1).T @ self.bnlast(x2)
        c.div_(N)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(1/32)
        off_diag = off_diagonal(c).pow_(2).sum().mul(1/32)
        loss = on_diag + 1/80 * off_diag
        
        return loss
        
        
class MySTTransformer(nn.Module):
    def __init__(self,
                 channel,
                 num_class,
                 window_size,
                 num_point,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 attention_3,
                 relative,
                 kernel_temporal,
                 double_channel,
                 drop_connect,
                 concat_original,
                 dv,
                 dk,
                 Nh,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 all_layers,
                 data_normalization,
                 visualization,
                 skip_conn,
                 adjacency,
                 bn_flag,
                 weight_matrix,
                 device,
                 n,
                 more_channels,
                 num_person=1,
                 use_data_bn=False,
                 backbone_config=None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5,
                 agcn = True):
        super(MySTTransformer, self).__init__()
        self.net = STTransformer(channel,
                 num_class,
                 window_size,
                 num_point,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 attention_3,
                 relative,
                 kernel_temporal,
                 double_channel,
                 drop_connect,
                 concat_original,
                 dv,
                 dk,
                 Nh,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 all_layers,
                 data_normalization,
                 visualization,
                 skip_conn,
                 adjacency,
                 bn_flag,
                 weight_matrix,
                 device,
                 n,
                 more_channels,
                 num_person=1,
                 use_data_bn=False,
                 backbone_config=None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5,
                 agcn = True)
                 
        #self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.net = torch.nn.DataParallel(self.net)

    def forward(self, x):
        return self.net(x)
    
class TCN_GCN_unit(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 A,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 relative,
                 device,
                 attention_3,
                 dv,
                 dk,
                 Nh,
                 num,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 num_point,
                 weight_matrix,
                 more_channels,
                 drop_connect,
                 starting_ch,
                 all_layers,
                 adjacency,
                 data_normalization,
                 visualization,
                 skip_conn,
                 layer=0,
                 kernel_size=9,
                 stride=1,
                 dropout=0.5,
                 use_local_bn=False,
                 mask_learning=False,
                 last=False,
                 last_graph=False,
                 agcn = False
                 ):
        super(TCN_GCN_unit, self).__init__()
        half_out_channel = out_channel / 2
        self.A = A
        self.V = A.size()[-1]
        self.C = in_channel
        self.last = last
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.num_point = num_point
        self.adjacency = adjacency
        self.last_graph = last_graph
        self.layer = layer
        self.stride = stride
        self.drop_connect = drop_connect
        self.visualization = visualization
        self.device = device
        self.all_layers = all_layers
        self.more_channels = more_channels

        if (out_channel >= starting_ch and attention or (self.all_layers and attention)):

            self.gcn1 = gcn_unit_attention(in_channel, out_channel, dv_factor=dv, dk_factor=dk, Nh=Nh,
                                           complete=True,
                                           relative=relative, only_attention=only_attention, layer=layer, incidence=A,
                                           bn_flag=True, last_graph=self.last_graph, more_channels=self.more_channels,
                                           drop_connect=self.drop_connect, adjacency=self.adjacency, num=num,
                                           data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                           visualization=self.visualization, num_point=self.num_point)
        else:

            if not agcn:
                self.gcn1 = unit_gcn(
                    in_channel,
                    out_channel,
                    A,
                    use_local_bn=use_local_bn,
                    mask_learning=mask_learning)
            else:
                self.gcn1 = unit_agcn(
                    in_channel,
                    out_channel,
                    A,
                    use_local_bn=use_local_bn,
                    mask_learning=mask_learning)

        if (out_channel >= starting_ch and tcn_attention or (self.all_layers and tcn_attention)):

            if out_channel <= starting_ch and self.all_layers:
                self.tcn1 = tcn_unit_attention_block(out_channel, out_channel, dv_factor=dv,
                                                     dk_factor=dk, Nh=Nh,
                                                     relative=relative, only_temporal_attention=only_temporal_attention,
                                                     dropout=dropout,
                                                     kernel_size_temporal=9, stride=stride,
                                                     weight_matrix=weight_matrix, bn_flag=True, last=self.last,
                                                     layer=layer,
                                                     device=self.device, more_channels=self.more_channels,
                                                     drop_connect=self.drop_connect, n=num,
                                                     data_normalization=self.data_normalization,
                                                     skip_conn=self.skip_conn,
                                                     visualization=self.visualization, dim_block1=dim_block1,
                                                     dim_block2=dim_block2, dim_block3=dim_block3, num_point=self.num_point)
            else:
                self.tcn1 = tcn_unit_attention(out_channel, out_channel, dv_factor=dv,
                                               dk_factor=dk, Nh=Nh,
                                               relative=relative, only_temporal_attention=only_temporal_attention,
                                               dropout=dropout,
                                               kernel_size_temporal=9, stride=stride,
                                               weight_matrix=weight_matrix, bn_flag=True, last=self.last,
                                               layer=layer,
                                               device=self.device, more_channels=self.more_channels,
                                               drop_connect=self.drop_connect, n=num,
                                               data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                               visualization=self.visualization, num_point=self.num_point)



        else:
            self.tcn1 = Unit2D(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                dropout=dropout,
                stride=stride)
        if ((in_channel != out_channel) or (stride != 1)):
            self.down1 = Unit2D(in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + (x if (self.down1 is None) else self.down1(x))
        return x


class TCN_GCN_unit_multiscale(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 kernel_size=9,
                 stride=1,
                 **kwargs):
        super(TCN_GCN_unit_multiscale, self).__init__()
        self.unit_1 = TCN_GCN_unit(
            in_channels,
            out_channels / 2,
            A,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs)
        self.unit_2 = TCN_GCN_unit(
            in_channels,
            out_channels - out_channels / 2,
            A,
            kernel_size=kernel_size * 2 - 1,
            stride=stride,
            **kwargs)

    def forward(self, x):
        return torch.cat((self.unit_1(x), self.unit_2(x)), dim=1)
        
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

def _transform_s(inputs1, mat):
    a1,a2 = inputs1.split([75,75],dim=2)
    b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25 = a1.chunk(25,dim=2)
    inputs1 = torch.stack([b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25],dim=3)
    inputs1 = inputs1.permute(0,2,1,3)
    rot = mat[:,0:3]
    trans = mat[:,3:6]
    inputs1 = inputs1.contiguous().view(-1,inputs1.size()[1],inputs1.size()[2]*inputs1.size()[3])
    trans, rot = _trans_rot(trans, rot)
    inputs1 = torch.cat((inputs1, Variable(inputs1.data.new(inputs1.size()[0],1,inputs1.size()[2]).fill_(1))), dim=1).cuda()
    inputs1 = torch.matmul(trans, inputs1)
    inputs1 = torch.matmul(rot, inputs1)
    inputs1 = inputs1.contiguous().view(-1,3,300,25)
    
    b31,b32,b33,b34,b35,b36,b37,b38,b39,b40,b41,b42,b43,b44,b45,b46,b47,b48,b49,b50,b51,b52,b53,b54,b55 = a2.chunk(25,dim=2)
    inputs2 = torch.stack([b31,b32,b33,b34,b35,b36,b37,b38,b39,b40,b41,b42,b43,b44,b45,b46,b47,b48,b49,b50,b51,b52,b53,b54,b55],dim=3)
    inputs2 = inputs2.permute(0,2,1,3)
    rot = mat[:,0:3]
    trans = mat[:,3:6]
    inputs2 = inputs2.contiguous().view(-1,inputs2.size()[1],inputs2.size()[2]*inputs2.size()[3])
    trans, rot = _trans_rot(trans, rot)
    inputs2 = torch.cat((inputs2, Variable(inputs2.data.new(inputs2.size()[0],1,inputs2.size()[2]).fill_(1))), dim=1).cuda()
    inputs2 = torch.matmul(trans, inputs2)
    inputs2 = torch.matmul(rot, inputs2)
    inputs2 = inputs2.contiguous().view(-1,3,300,25)
    
    inputs = torch.stack([inputs1,inputs2],dim=4)
    return inputs

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
    inputs2 = inputs2.contiguous().view(-1, inputs2.size()[1], inputs2.size()[2] * inputs2.size()[3])
    inputs2 = torch.cat((inputs2, Variable(inputs2.data.new(inputs2.size()[0], 1, inputs2.size()[2]).fill_(1))), dim=1).cuda()
    inputs2 = torch.matmul(trans, inputs2)
    inputs2 = torch.matmul(rot, inputs2)
    inputs2 = inputs2.contiguous().view(-1, 3, 300, 25)

    inputs = torch.stack([inputs1, inputs2], dim=4)

    return inputs, view


def _transform_shape(inputs1):
    a1, a2 = inputs1.split([75, 75], dim=2)
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25 = a1.chunk(25, dim=2)
    inputs1 = torch.stack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25], dim=3)
    inputs1 = inputs1.permute(0, 2, 1, 3)
    inputs1 = inputs1.contiguous().view(-1, 3, 300, 25)
    b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55 = a2.chunk(25, dim=2)
    inputs2 = torch.stack([b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46, b47, b48, b49, b50, b51, b52, b53, b54, b55], dim=3)
    inputs2 = inputs2.permute(0, 2, 1, 3)
    inputs2 = inputs2.contiguous().view(-1, 3, 300, 25)
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