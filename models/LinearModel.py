from __future__ import print_function
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class LinearClassifierAlexNet(nn.Module):
    def __init__(self, layer=5, n_label=1000, pool_type='max'):
        super(LinearClassifierAlexNet, self).__init__()
        if layer == 1:
            pool_size = 10
            nChannels = 96
        elif layer == 2:
            pool_size = 6
            nChannels = 256
        elif layer == 3:
            pool_size = 5
            nChannels = 384
        elif layer == 4:
            pool_size = 5
            nChannels = 384
        elif layer == 5:
            pool_size = 6
            nChannels = 256
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.classifier = nn.Sequential()

        if layer < 5:
            if pool_type == 'max':
                self.classifier.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.classifier.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.classifier.add_module('Flatten', Flatten())
        self.classifier.add_module('LinearClassifier', nn.Linear(nChannels*pool_size*pool_size, n_label))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)


class LinearClassifierResNet(nn.Module):
    def __init__(self, layer=6, n_label=1000, pool_type='avg', width=1):
        super(LinearClassifierResNet, self).__init__()
        if layer == 1:
            pool_size = 8
            nChannels = 128 * width
            pool = pool_type
        elif layer == 2:
            pool_size = 6
            nChannels = 256 * width
            pool = pool_type
        elif layer == 3:
            pool_size = 4
            nChannels = 512 * width
            pool = pool_type
        elif layer == 4:
            pool_size = 3
            nChannels = 1024 * width
            pool = pool_type
        elif layer == 5:
            pool_size = 7
            nChannels = 2048 * width
            pool = pool_type
        elif layer == 6:
            pool_size = 1
            nChannels = 2048 * width
            pool = pool_type
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.classifier = nn.Sequential()
        if layer < 5:
            if pool == 'max':
                self.classifier.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool == 'avg':
                self.classifier.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        else:
            pass

        self.classifier.add_module('Flatten', Flatten())
        print('classifier input: {}'.format(nChannels * pool_size * pool_size))
        self.classifier.add_module('LiniearClassifier', nn.Linear(nChannels * pool_size * pool_size, n_label))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)


class LinearClassifierMsg3d(nn.Module):
    def __init__(self, n_label=40):
        super(LinearClassifierMsg3d, self).__init__()
        self.c = nn.Linear(384,n_label)

    def forward(self, x):
        x = self.c(x)
        return x

       
class LinearClassifierCNN(nn.Module):
    def __init__(self, n_label=60):
        super(LinearClassifierCNN, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256,n_label)
        
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
                m.momentum = 0.99
                m.eps = 1e-3

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

 
class LinearClassifierTransformer(nn.Module):
    def __init__(self, n_label=60):
        super(LinearClassifierTransformer, self).__init__()
        self.fc1 = nn.Linear(512, n_label)
       
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
                m.momentum = 0.99
                m.eps = 1e-3

    def forward(self, x):
        x = self.fc1(x)
        return x

        
class LinearClassifierCtrgcn(nn.Module):
    def __init__(self, n_label=40):
        super(LinearClassifierCtrgcn, self).__init__()
        self.c = nn.Linear(100,n_label)
        nn.init.normal_(self.c.weight, 0, math.sqrt(2. / n_label))

    def forward(self, x):
        x = self.c(x)
        return x
