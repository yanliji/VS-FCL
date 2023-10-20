# 这个是train_stage2.py
from __future__ import print_function
import os
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import tensorboard_logger as tb_logger
from util import adjust_learning_rate, AverageMeter
from dataset_new_stage2 import NTUDataLoaders
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC
from models.msg3d1 import MyMsg3d
from models.ctrgcn1 import MyCTRGCN
from models.sttr1 import MySTTransformer
from apex import amp, optimizers

import sys
import socket
import random
import math
from torchvision import transforms, datasets
#from dataset import RGB2Lab, RGB2YCbCr
#from dataset import NTUDataLoaders
import torch.nn.functional as F
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss
#from dataset import ImageFolderInstance

# model distributed training setting
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '5678'


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet', 'resnet',
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
    parser.add_argument('--case', type=int, default=0, choices=[0,1,2,3,4,5,6,7,8])

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr', 'Multiview'])

    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--state', default='train', type=str, choices=['train','test'])
    parser.add_argument('--monitor', type=str, default='val_loss', help='quantity to monitor (default: val_acc)')

    opt = parser.parse_args()
    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = 'stage2_lossquan_memory_{}_{}_{}_lr_{}_decay_{}_bsz_{}_case_{}_dataset_{}'.format(opt.method, opt.nce_k, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.case, opt.dataset)
    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)
    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)
    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)
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
    pred = pred.view(batch_size,1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    return correct.mul_(100.0 / batch_size)


def get_train_loader(args):
    """get the train loader"""

    ntu_loaders = NTUDataLoaders(args.dataset, args.case)
    train_loader = ntu_loaders.get_train_loader(args.batch_size, args.num_workers)
    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.num_workers)
    n_data = ntu_loaders.get_train_size()

    print('number of train: {}'.format(ntu_loaders.get_train_size()))
    print('number of val: {}'.format(ntu_loaders.get_val_size()))

    return train_loader, val_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        model = MyAlexNetCMC(args.feat_dim)
    elif args.model.startswith('resnet'):
        model = MyResNetsCMC(num_classes = 60)
    elif args.model == 'msg3d':
        model = MyMsg3d(40,25,2,13,6,'graph.ntu_rgb_d.AdjMatrixGraph')
    elif args.model == 'sttr':
        model = MySTTransformer(num_class=60,channel=3,window_size=300,num_point=25,num_person=2,mask_learning=True,use_data_bn=True,attention=True,only_attention=False,tcn_attention=True,data_normalization=True,skip_conn=True,weight_matrix=2,only_temporal_attention=False,bn_flag=True,attention_3=False,kernel_temporal=9,more_channels=False,double_channel=True,drop_connect=True,concat_original=True,all_layers=False,adjacency=False,agcn=True,dv=0.25,dk=0.25,Nh=8,n=4,dim_block1=10,dim_block2=30,dim_block3=75,relative=False,graph='graph.ntu_rgb_d.Graph',visualization=False,device=[0,1,2,3])
    elif args.model == 'ctrgcn':
        model = MyCTRGCN(40,25,2,'graph.ntu_rgb_d.AdjMatrixGraph',3,0,True)
    # elif args.model == 'dstanet':
    #     model = MyDSTANet()
    #     classifier = MyDSTANet1()
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    criterion = nn.CrossEntropyLoss().cuda()
    criterion1 = nn.MSELoss().cuda()
    criterion2 = nn.MSELoss().cuda()
    criterion3 = nn.MultiLabelSoftMarginLoss(reduction='mean').cuda()

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    return model, criterion, criterion1, criterion2, criterion3


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #optimizer = LARS(filter(lambda p: p.requires_grad, model.parameters()), lr=0, weight_decay=args.weight_decay, weight_decay_filter=exclude_bias_and_norm, lars_adaptation_filter=exclude_bias_and_norm)
    return optimizer


def train(epoch, train_loader, model, criterion, criterion1, criterion2, criterion3, optimizer, opt):
    """
    one epoch training
    """
    model.train()
    #classifier.train()
    #contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy1_meter = AverageMeter()
    accuracy2_meter = AverageMeter()
    accuracy3_meter = AverageMeter()
    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()


    end = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for step, (inputs, target, viewt, index) in enumerate(train_loader):
        if inputs.size(0) != opt.batch_size:
            break

        data_time.update(time.time() - end)
        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            index = index.cuda()
            inputs = inputs.cuda()
            target = target.cuda()
            viewt = viewt.cuda()


        # ===================forward=====================
        # the fisrt V-FCL training stage outputs view-specific, view-common features, with view discrimination vector
        #fs, fc, fv = model(inputs, viewt)
        # the second S-FCL training stage outputs view-common features, with semantic discrimination vector (original main view and augmented assistant view)
        outp, outp1, fc1, fc2 = model(inputs, viewt)
        loss1 = (criterion(fc1,target) + criterion(fc2,target))/2 # 
        acc1 = (accuracy(fc1,target) + accuracy(fc2,target))/2 # 

        # the current version of CV-FCL, the semantic-term S-FCL, UESTC dataset with 40 actions 
        list0,list1,list2,list3,list4,list5,list6,list7,list8,list9,list10,list11,list12,list13,list14,list15,list16,list17,list18,list19,list20,list21,list22,list23,list24,list25,list26,list27,list28,list29,list30,list31,list32,list33,list34,list35,list36,list37,list38,list39 = list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list()
        for i in range(bsz):
            if target[i] == 0:
                list0.append(outp[i,:].view(1,128)) # feature dim 128
                list0.append(outp1[i,:].view(1,128))
            if target[i] == 1:
                list1.append(outp[i,:].view(1,128))
                list1.append(outp1[i,:].view(1,128))
            if target[i] == 2:
                list2.append(outp[i,:].view(1,128))
                list2.append(outp1[i,:].view(1,128))
            if target[i] == 3:
                list3.append(outp[i,:].view(1,128))
                list3.append(outp1[i,:].view(1,128))
            if target[i] == 4:
                list4.append(outp[i,:].view(1,128))
                list4.append(outp1[i,:].view(1,128))
            if target[i] == 5:
                list5.append(outp[i,:].view(1,128))
                list5.append(outp1[i,:].view(1,128))
            if target[i] == 6:
                list6.append(outp[i,:].view(1,128))
                list6.append(outp1[i,:].view(1,128))
            if target[i] == 7:
                list7.append(outp[i,:].view(1,128))
                list7.append(outp1[i,:].view(1,128))
            if target[i] == 8:
                list8.append(outp[i,:].view(1,128))
                list8.append(outp1[i,:].view(1,128))
            if target[i] == 9:
                list9.append(outp[i,:].view(1,128))
                list9.append(outp1[i,:].view(1,128))
            if target[i] == 10:
                list10.append(outp[i,:].view(1,128))
                list10.append(outp1[i,:].view(1,128))
            if target[i] == 11:
                list11.append(outp[i,:].view(1,128))
                list11.append(outp1[i,:].view(1,128))
            if target[i] == 12:
                list12.append(outp[i,:].view(1,128))
                list12.append(outp1[i,:].view(1,128))
            if target[i] == 13:
                list13.append(outp[i,:].view(1,128))
                list13.append(outp1[i,:].view(1,128))
            if target[i] == 14:
                list14.append(outp[i,:].view(1,128))
                list14.append(outp1[i,:].view(1,128))
            if target[i] == 15:
                list15.append(outp[i,:].view(1,128))
                list15.append(outp1[i,:].view(1,128))
            if target[i] == 16:
                list16.append(outp[i,:].view(1,128))
                list16.append(outp1[i,:].view(1,128))
            if target[i] == 17:
                list17.append(outp[i,:].view(1,128))
                list17.append(outp1[i,:].view(1,128))
            if target[i] == 18:
                list18.append(outp[i,:].view(1,128))
                list18.append(outp1[i,:].view(1,128))
            if target[i] == 19:
                list19.append(outp[i,:].view(1,128))
                list19.append(outp1[i,:].view(1,128))
            if target[i] == 20:
                list20.append(outp[i,:].view(1,128))
                list20.append(outp1[i,:].view(1,128))
            if target[i] == 21:
                list21.append(outp[i,:].view(1,128))
                list21.append(outp1[i,:].view(1,128))
            if target[i] == 22:
                list22.append(outp[i,:].view(1,128))
                list22.append(outp1[i,:].view(1,128))
            if target[i] == 23:
                list23.append(outp[i,:].view(1,128))
                list23.append(outp1[i,:].view(1,128))
            if target[i] == 24:
                list24.append(outp[i,:].view(1,128))
                list24.append(outp1[i,:].view(1,128))
            if target[i] == 25:
                list25.append(outp[i,:].view(1,128))
                list25.append(outp1[i,:].view(1,128))
            if target[i] == 26:
                list26.append(outp[i,:].view(1,128))
                list26.append(outp1[i,:].view(1,128))
            if target[i] == 27:
                list27.append(outp[i,:].view(1,128))
                list27.append(outp1[i,:].view(1,128))
            if target[i] == 28:
                list28.append(outp[i,:].view(1,128))
                list28.append(outp1[i,:].view(1,128))
            if target[i] == 29:
                list29.append(outp[i,:].view(1,128))
                list29.append(outp1[i,:].view(1,128))
            if target[i] == 30:
                list30.append(outp[i,:].view(1,128))
                list30.append(outp1[i,:].view(1,128))
            if target[i] == 31:
                list31.append(outp[i,:].view(1,128))
                list31.append(outp1[i,:].view(1,128))
            if target[i] == 32:
                list32.append(outp[i,:].view(1,128))
                list32.append(outp1[i,:].view(1,128))
            if target[i] == 33:
                list33.append(outp[i,:].view(1,128))
                list33.append(outp1[i,:].view(1,128))
            if target[i] == 34:
                list34.append(outp[i,:].view(1,128))
                list34.append(outp1[i,:].view(1,128))
            if target[i] == 35:
                list35.append(outp[i,:].view(1,128))
                list35.append(outp1[i,:].view(1,128))
            if target[i] == 36:
                list36.append(outp[i,:].view(1,128))
                list36.append(outp1[i,:].view(1,128))
            if target[i] == 37:
                list37.append(outp[i,:].view(1,128))
                list37.append(outp1[i,:].view(1,128))
            if target[i] == 38:
                list38.append(outp[i,:].view(1,128))
                list38.append(outp1[i,:].view(1,128))
            if target[i] == 39:
                list39.append(outp[i,:].view(1,128))
                list39.append(outp1[i,:].view(1,128))

        # compute the mean center of each action cluster
        for ll in list([list0,list1,list2,list3,list4,list5,list6,list7,list8,list9,list10,list11,list12,list13,list14,list15,list16,list17,list18,list19,list20,list21,list22,list23,list24,list25,list26,list27,list28,list29,list30,list31,list32,list33,list34,list35,list36,list37,list38,list39]):
            if len(ll) > 1:
                ele = ll[0]
                for x in ll[1:]:
                    x = torch.cat((ele,x),dim=0)
                c = torch.mean(x,dim=0).view(1,128)
                ll.append(c)

        clist = list() # center list
        llist = list() # feature list
        clist1 = list()
        llist1 = list()


        for i,ll in enumerate(list([list0,list1,list2,list3,list4,list5,list6,list7,list8,list9,list10,list11,list12,list13,list14,list15,list16,list17,list18,list19,list20,list21,list22,list23,list24,list25,list26,list27,list28,list29,list30,list31,list32,list33,list34,list35,list36,list37,list38,list39])):
            if (len(ll) > 1) & (len(ll) <= 8):
                clist.append(ll[-1])
                llist.append(ll)

        loss2 = torch.zeros(1,requires_grad=True).cuda()

        for i,ll in enumerate(llist):
            fsw = list()
            label = list()
            for x in ll[:-1]:
                # view-common feature with corresponding action cluster center, Sw equation, multi-classification label 0, temperature coefficient 0.06
                fsw.append(torch.abs((x-ll[-1]).t().mm(x-ll[-1])).trace().item()/0.06)
                label.append(0)
            for j,ll1 in enumerate(clist):
                if j != i:
                    # different view-common action cluster centers, Sb equation, multi-classification label 1, temperature coefficient 0.06
                    fsw.append(torch.abs((clist[i]-clist[j]).t().mm(clist[i]-clist[j])).trace().item()/0.06)
                    label.append(1)

            fsw = torch.tensor(fsw).cuda().view(1,-1)
            label = torch.tensor(label).cuda().view(1,-1)
            loss2 += criterion3(fsw,label)

        loss2 = 0.1*loss2 # 
        loss = loss1 + loss2


        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()


        # ===================meters=====================
        losses.update(loss.item(), bsz)
        loss1_meter.update(loss1.item(), bsz)
        loss2_meter.update(loss2.item(), bsz)
        accuracy1_meter.update(acc1[0].item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (step + 1) % opt.print_freq == 0:
            print('Stage2 Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss1 {loss1.val:.3f} ({loss1.avg:.3f})\t'
                  'loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t'
                  'acc1 {acc1.val:.3f} ({acc1.avg:.3f})\t'.format(
                epoch, step + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss1=loss1_meter, loss2=loss2_meter, acc1=accuracy1_meter))

    return losses.avg, loss1_meter.avg, loss2_meter.avg, accuracy1_meter.avg


def validate(epoch, val_loader, model, criterion, criterion1, criterion2, criterion3, optimizer, opt):
    """
    evaluation
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy1_meter = AverageMeter()
    accuracy2_meter = AverageMeter()
    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #contrast.eval()

    end = time.time()
    # load data, action label, view label, index
    for idx, (input, target, viewt, index) in enumerate(val_loader):
        if input.size(0) != opt.batch_size:
            break

        input = input.float()
        if torch.cuda.is_available():
            index = index.cuda()
            input = input.cuda()
            target = target.cuda()
            viewt = viewt.cuda()

        bsz = input.size(0)


        with torch.no_grad():
            outp, outp1, fc1, fc2 = model(input, viewt) # 
            loss = (criterion(fc1,target) + criterion(fc2,target))/2
            acc = (accuracy(fc1,target) + accuracy(fc2,target))/2


        loss1_meter.update(loss.item(), bsz)
        accuracy1_meter.update(acc[0].item(), bsz)


    return loss1_meter.avg, accuracy1_meter.avg



# LARS optimizer   
class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001, weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta, weight_decay_filter=weight_decay_filter, lars_adaptation_filter=lars_adaptation_filter)
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
                    q = torch.where(param_norm > 0., torch.where(update_norm > 0, (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, val_loader, n_data= get_train_loader(args)

    # set the model
    model, criterion, criterion1, criterion2, criterion3 = set_model(args, n_data)
    
    # set the optimizer
    optimizer = set_optimizer(args, model)
    
    # set mixed precision
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    
    '''
    # model distributed training setting
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")  
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    '''

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            #contrast.load_state_dict(checkpoint['contrast'])
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

    # early stopping setting
    if args.monitor == 'val_acc':
        mode = 'max'
        monitor_op = np.greater
        best = -np.Inf
        str_op = 'improve'
    elif args.monitor == 'val_loss':
        mode = 'min'
        monitor_op = np.less
        best = np.Inf
        str_op = 'reduce'
    
    if args.dataset=='NTU':
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=args.lr_decay_rate,
                                  patience=5, cooldown=3, verbose=True)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=args.lr_decay_rate,
                                      patience=5, cooldown=3, verbose=True)



    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        loss,loss1,loss2,acc1 = train(epoch, train_loader, model, criterion, criterion1, criterion2, criterion3, optimizer, args)
        time2 = time.time()

        loss1v, acc1v = validate(epoch, val_loader, model, criterion, criterion1, criterion2, criterion3, optimizer, args)
        print('val_loss1 {:.2f}, val_acc1 {:.2f}'.format(loss1v, acc1v))

        print('epoch {}, total time {:.2f}\t'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('stage2_val_loss1', loss1v, epoch) # validate
        logger.log_value('stage2_val_acc1', acc1v, epoch)
        
        logger.log_value('stage2_loss1', loss1, epoch) # train
        logger.log_value('stage2_loss', loss, epoch)
        logger.log_value('stage2_loss2', loss2, epoch)
        logger.log_value('stage2_acc1', acc1, epoch)
        
        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'monitor': args.monitor,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state



if __name__ == '__main__':
    main()
