import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .net import conv_init


class unit_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 use_local_bn=False,
                 kernel_size=1,
                 stride=1,
                 mask_learning=False):
        super(unit_gcn, self).__init__()

        self.V = A.size()[-1]
        self.A = Variable(A.clone(), requires_grad=False).view(-1, self.V, self.V)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_learning = mask_learning
        self.num_A = self.A.size()[0]
        self.use_local_bn = use_local_bn

        self.conv_list = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1)) for i in range(self.num_A)
        ])
        if mask_learning:
            self.mask = nn.Parameter(torch.ones(self.A.size()))
        if use_local_bn:
            self.bn = nn.BatchNorm1d(self.out_channels * self.V)
        else:
            self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

        for conv in self.conv_list:
            conv_init(conv)

    def forward(self, x):
        N, C, T, V = x.size()
        self.A = self.A.cuda(x.get_device())
        A = self.A

        if self.mask_learning:
            A *=  self.mask

        for i, a in enumerate(A):
            xa = x.reshape(-1, V).mm(a).reshape(N, C, T, V)
            if i == 0:
                y = self.conv_list[i](xa)
            else:
                y +=  self.conv_list[i](xa)

        if self.use_local_bn:
            y = y.permute(0, 1, 3, 2).contiguous().view(
                N, self.out_channels * V, T)
            y = self.bn(y)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else:
            y = self.bn(y)

        y = self.relu(y)

        return y
