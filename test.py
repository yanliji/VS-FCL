
from __future__ import print_function
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket
import random
import numpy as np
import torch.nn.functional as F
from math import exp
import tensorboard_logger as tb_logger
from util import adjust_learning_rate, AverageMeter
from dataset_new import NTUDataLoaders
import torch.nn as nn
from torch import optim
from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC
from models.msg3d1 import MyMsg3d
from models.ctrgcn1 import MyCTRGCN
from models.sttr1 import MySTTransformer
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from torchvision import transforms, datasets
#from dataset import RGB2Lab, RGB2YCbCr
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#from dataset import ImageFolderInstance

try:
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='20,40,60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3', 'msg3d', 'sttr', 'ctrgcn','dstanet'])
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet', 'NTU', 'NTU 120', 'UESTC'])
    parser.add_argument('--case', type=int, default=0, choices=[0,1,2,3,4,5,6,7])

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr', 'Multiview'])

    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--state', default='train', type=str, choices=['train','test'])

    opt = parser.parse_args()
    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = 'Test_stage2_lossquan_memory_{}_{}_{}_lr_{}_decay_{}_bsz_{}_case_{}_dataset_{}'.format(opt.method, opt.nce_k, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.case, opt.dataset)
    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)
    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if opt.dataset == 'imagenet100':
        opt.n_label = 100
    if opt.dataset == 'imagenet':
        opt.n_label = 1000
    if opt.dataset == 'NTU':
        opt.n_label = 60
    if opt.dataset == 'NTU 120':
        opt.n_label = 120
    if opt.dataset == 'UESTC':
        opt.n_label = 40
    
    return opt


def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.view(8,1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    return correct.mul_(100.0 / batch_size)


def get_train_loader(args):

    ntu_loaders = NTUDataLoaders(args.dataset, args.case)
    test_loader = ntu_loaders.get_test_loader(args.batch_size, args.num_workers)
    n_data = ntu_loaders.get_test_size()

    print('number of test: {}'.format(ntu_loaders.get_test_size()))
    print('number of val: {}'.format(ntu_loaders.get_val_size()))

    return test_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        model = MyAlexNetCMC(args.feat_dim)
    elif args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
    elif args.model == 'msg3d':
        model = MyMsg3d(40,25,2,13,6,'graph.ntu_rgb_d.AdjMatrixGraph')
    elif args.model == 'sttr':
        model = MySTTransformer(num_class=60,channel=3,window_size=300,num_point=25,num_person=2,mask_learning=True,use_data_bn=True,attention=True,only_attention=True,tcn_attention=True,data_normalization=True,skip_conn=True,weight_matrix=2,only_temporal_attention=False,bn_flag=True,attention_3=False,kernel_temporal=9,more_channels=False,double_channel=True,drop_connect=True,concat_original=True,all_layers=False,adjacency=False,agcn=False,dv=0.25,dk=0.25,Nh=8,n=4,dim_block1=10,dim_block2=30,dim_block3=75,relative=False,graph='graph.ntu_rgb_d.Graph',visualization=False,device=[0,1])
        #for name, value in model.named_parameters():
            #print(name)
            #print(value)
    elif args.model == 'ctrgcn':
        model = MyCTRGCN(40,25,2,'graph.ntu_rgb_d.AdjMatrixGraph',3,0,True)
    #elif args.model == 'dstanet':
    #    model = MyDSTANet()
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion1 = nn.CrossEntropyLoss().cuda()
    criterion2 = nn.MSELoss().cuda()

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion_ab, criterion_l, criterion1, criterion2


def set_optimizer(args, model):
    # return optimizer
    
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return optimizer

def test(epoch,test_loader, model, criterion2, opt):
    accuracies = AverageMeter()
    losses = AverageMeter()

    #checkpoint1 = opt.model_path
    #checkpoint2 = opt.model_path
    model.eval()
    #classifier.eval()
    preds, label = list(), list()
    x1, y1, z1 = list(), list(), list()
    trans1, viewlabel = list(), list()
    y2,y3,y4,y5,y6,y7,y8,y9 = list(), list(), list(), list(), list(), list(), list(), list()
    x2,x3,x4,x5,x6,x7,x8,x9 = list(), list(), list(), list(), list(), list(), list(), list()

    '''
    ckpt = torch.load(opt.model_path)
    new_dict = {}
    for k,v in ckpt['model'].items():
        if ('encoderp_s' in k) | ('transp_s' in k) | ('decoder' in k) | ('data_bn' in k):
            new_dict[k] = v
    model.load_state_dict(new_dict)
    '''
    model.load_state_dict(torch.load(opt.model_path)['model'], strict=False)
    #classifier.load_state_dict(torch.load(checkpoint2)['model'], strict=False)

    with torch.no_grad():
        count = 0
        for idx, (input, target, viewt, index) in enumerate(test_loader):
            if input.size(0) != opt.batch_size:
                break
            bsz = input.size(0)
            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)
            index = index.cuda(opt.gpu, non_blocking=True)
            viewt = viewt.cuda(opt.gpu, non_blocking=True)

            # compute output
            p, p1, output, output1 = model(input, viewt) # 第二阶段的

            '''
            SSIM criterion for skeleton reconstruction
            cri = SSIM().cuda()
            for i,t in enumerate(list(viewt)[0:bsz]):
                if t == 0:
                    c = c+1
                    loss = loss + cri(out[i,:,0:200,:].unsqueeze(0), out_gt[i,:,0:200,:].unsqueeze(0))
            loss = loss/c
            '''
            
            pred = output.cpu().data.numpy()
            target = target.cpu().numpy()
            preds = preds + list(pred)
            label = label + list(target)

            #viewt = viewt.cpu().numpy()
            #viewlabel = viewlabel + list(viewt)
            #trans = trans.cpu().data.numpy()
            #trans1 = trans1 + list(trans)
            #pred_label = np.argmax(pred, axis=-1)
            #acc = accuracy(output, target)
            #accuracies.update(acc[0], input.size(0))
            
            '''
            # same-action different-view view-specific feature, batch size 64, action category 21
            if target.equal(torch.full((64,1),21).view(64).cuda()):
                count = count + 1
                tsne = TSNE(n_components=2)
                x = outv.cpu().data.numpy()
                x1 = x1 + list(x)
                y = viewt.cpu().numpy()
                y1 = y1 + list(y)
                #y = target.cpu().numpy()
                #y1 = y1 + list(y)
                #z = viewt.cpu().numpy()
                #z1 = z1 + list(z)
            # different-action view common feature  
            tsne = TSNE(n_components=2)
            x = outvp.cpu().data.numpy()
            x1 = x1 + list(x)
            y = target.cpu().numpy()
            y1 = y1 + list(y)
            z = viewt.cpu().numpy()
            z1 = z1 + list(z)
            '''

    #viewlabel = np.array(viewlabel)
    #trans1 = np.array(trans1)
    #np.savetxt('trans.txt', trans1, fmt='%f', delimiter=',')
    #np.savetxt('viewlabels.txt', viewlabel, fmt='%f', delimiter=',')

    preds = np.array(preds)
    label = np.array(label)
    preds_label = np.argmax(preds, axis=-1)
    total = ((label - preds_label) == 0).sum()
    total = float(total)

    '''
    # respectively draw 8 views for 40 actions in the UESTC dataset
    x_tsne = tsne.fit_transform(x1)
    x1 = x_tsne
    
    l2 = [idx for idx,i in enumerate(z1) if i == 0]
    for x in l2:
        if np.random.randint(10)==0:
            y2.append(y1[x])
            x2.append(x1[x,:])
    
    l3 = [idx for idx,i in enumerate(z1) if i == 1]
    for x in l3:
        if np.random.randint(5)==0:
            y3.append(y1[x])
            x3.append(x1[x,:])
    
    l4 = [idx for idx,i in enumerate(z1) if i == 2]
    for x in l4:
        if np.random.randint(10)==0:
            y4.append(y1[x])
            x4.append(x1[x,:])
    
    l5 = [idx for idx,i in enumerate(z1) if i == 3]
    for x in l5:
        if np.random.randint(5)==0:
            y5.append(y1[x])
            x5.append(x1[x,:])
    
    l6 = [idx for idx,i in enumerate(z1) if i == 4]
    for x in l6:
        if np.random.randint(10)==0:
            y6.append(y1[x])
            x6.append(x1[x,:])
    
    l7 = [idx for idx,i in enumerate(z1) if i == 5]
    for x in l7:
        if np.random.randint(5)==0:
            y7.append(y1[x])
            x7.append(x1[x,:])
    
    l8 = [idx for idx,i in enumerate(z1) if i == 6]
    for x in l8:
        if np.random.randint(10)==0:
            y8.append(y1[x])
            x8.append(x1[x,:])
    
    l9 = [idx for idx,i in enumerate(z1) if i == 7]
    for x in l9:
        if np.random.randint(5)==0:
            y9.append(y1[x])
            x9.append(x1[x,:])
    
    x_tsne2 = np.array(x2)
    #x_tsne3 = np.array(x3)
    x_tsne4 = np.array(x4)
    #x_tsne5 = np.array(x5)
    x_tsne6 = np.array(x6)
    #x_tsne7 = np.array(x7)
    x_tsne8 = np.array(x8)
    #x_tsne9 = np.array(x9)

    s1 = plt.scatter(x_tsne2[:, 0], x_tsne2[:, 1], c=y2, marker='o', s=6)
    #s2 = plt.scatter(x_tsne3[:, 0], x_tsne3[:, 1], c=y3, marker='*', s=6)
    s3 = plt.scatter(x_tsne4[:, 0], x_tsne4[:, 1], c=y4, marker='+', s=6)
    #s4 = plt.scatter(x_tsne5[:, 0], x_tsne5[:, 1], c=y5, marker='^', s=6)
    s5 = plt.scatter(x_tsne6[:, 0], x_tsne6[:, 1], c=y6, marker='d', s=6)
    #s6 = plt.scatter(x_tsne7[:, 0], x_tsne7[:, 1], c=y7, marker='x', s=6)
    s7 = plt.scatter(x_tsne8[:, 0], x_tsne8[:, 1], c=y8, marker='1', s=6)
    #s8 = plt.scatter(x_tsne9[:, 0], x_tsne9[:, 1], c=y9, marker='|', s=6)
    '''

    #plt.scatter(x_tsne[:,0],x_tsne[:,1],c=y1,cmap='rainbow',marker='o',s=10) # draw view-specific feature
    # draw view-common feature based on s1-s8
    #plt.legend((s1,s2,s3,s4,s5,s6,s7,s8),('V0','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7'), loc = 1, markerscale=2.0)
    #plt.legend((s1,s3,s5,s7),('V0', 'V2', 'V4', 'V6'), loc = 1, markerscale=2.0)
    #plt.savefig('fs_case2_new.jpg')

    print(" Test Model Accuracy:%.2f" % (total / len(label) * 100))
    #print('val_loss {:.2f}'.format(losses.avg))
    
 
class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
 
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
 
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
 
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


def main():

    # parse the args
    args = parse_option()

    # set the loader
    test_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_ab, criterion_l, criterion1, criterion2 = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # set mixed precision
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    epoch=1
    test(epoch,test_loader,model,criterion2,args)
    print('Done Testing! epoch num = 1')


if __name__ == '__main__':
    main()
    print('Done Testing main!')
