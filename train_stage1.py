# train_stage1.py 
from __future__ import print_function
import os
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import tensorboard_logger as tb_logger
from util import adjust_learning_rate, AverageMeter
from dataset_new_stage1 import NTUDataLoaders
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
# from NCE.NCEAverage import NCEAverage
# from NCE.NCECriterion import NCECriterion
# from NCE.NCECriterion import NCESoftmaxLoss
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
    opt.model_name = 'memory_{}_{}_{}_lr_{}_decay_{}_bsz_{}_case_{}_dataset_{}'.format(opt.method, opt.nce_k, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.case, opt.dataset)
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

    # print model parameters
    #for name, value in model.named_parameters():
        #print(name)
        #print(value)
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
    loss3_meter = AverageMeter()
    loss4_meter = AverageMeter()
    loss5_meter = AverageMeter()

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
        # In the first stage, tensorBoard includes acc3, loss, loss1, loss2, and loss3.
        # In the second stage，tensorboard includes acc1、loss、loss1、loss2、val_acc1、val_loss1.
        # the first stage: the fisrt V-FCL training stage outputs view-specific, view-common features, with view discrimination vector

        fs, fc, fv = model(inputs, viewt) # 
        #the second stage: the second S-FCL training stage outputs view-common features, with semantic discrimination vector (original main view and augmented assistant view)
        # outp, outp1, fc1, fc2 = model(inputs, viewt)
        # loss1 = (criterion(fc1,target) + criterion(fc2,target))/2 # 
        # acc1 = (accuracy(fc1,target) + accuracy(fc2,target))/2 # 
        loss3 = criterion(fv,viewt) #
        acc3 = accuracy(fv,viewt) # 

        # view-term V-FCL, UESTC dataset   8 views      V-FCL loss
        list0,list1,list2,list3,list4,list5,list6,list7 = list(),list(),list(),list(),list(),list(),list(),list()
        for i in range(bsz):
            if viewt[i] == 0:
                list0.append(fs[i,:].view(1,128))
            if viewt[i] == 1:
                list1.append(fs[i,:].view(1,128))
            if viewt[i] == 2:
                list2.append(fs[i,:].view(1,128))
            if viewt[i] == 3:
                list3.append(fs[i,:].view(1,128))
            if viewt[i] == 4:
                list4.append(fs[i,:].view(1,128))
            if viewt[i] == 5:
                list5.append(fs[i,:].view(1,128))
            if viewt[i] == 6:
                list6.append(fs[i,:].view(1,128))
            if viewt[i] == 7:
                list7.append(fs[i,:].view(1,128))

        for ll in list([list0,list1,list2,list3,list4,list5,list6,list7]):
            if len(ll) > 1:
                ele = ll[0]
                for x in ll[1:]:
                    x = torch.cat((ele,x),dim=0)
                c = torch.mean(x,dim=0).view(1,128)
                ll.append(c)

        #sw = torch.zeros(128,128).cuda()
        #sb = torch.zeros(128,128).cuda()
        fcsw = torch.zeros(128,128,requires_grad=True).cuda()

        clist = list()
        llist = list()

        for i,ll in enumerate(list([list0,list1,list2,list3,list4,list5,list6,list7])):
            if len(ll) > 1:
                clist.append(ll[-1])
                llist.append(ll)

        loss2 = torch.zeros(1,requires_grad=True).cuda()

        for i,ll in enumerate(llist):
            fsw = list()
            label = list()
            for x in ll[:-1]:
                fsw.append(torch.abs((x-ll[-1]).t().mm(x-ll[-1])).trace().item()/0.06)
                label.append(0)
            for j,ll1 in enumerate(clist):
                if j != i:
                    fsw.append(torch.abs((clist[i]-clist[j]).t().mm(clist[i]-clist[j])).trace().item()/0.06)
                    label.append(1)

            fsw = torch.tensor(fsw).cuda().view(1,-1)
            label = torch.tensor(label).cuda().view(1,-1)
            loss2 += criterion3(fsw,label)

        for i in range(bsz):
            fcsw += (fc[i,:].view(1,128)-fc.mean(0).view(1,128)).t().mm(fc[i,:].view(1,128)-fc.mean(0).view(1,128))

        loss1 = torch.abs(fcsw).trace()
        loss1 = 0.0001*loss1
        loss2 = 0.0001*loss2
        loss = loss1 + loss2 + loss3  #


        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()


        # ===================meters=====================
        # the first stage: loss = loss1 + loss2 + loss3    the second stage: loss = loss1 + loss2
        
        losses.update(loss.item(), bsz)
        loss1_meter.update(loss1.item(), bsz)
        loss2_meter.update(loss2.item(), bsz)
        loss3_meter.update(loss3.item(), bsz)
        accuracy3_meter.update(acc3[0].item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()


        if (step + 1) % opt.print_freq == 0:
            print('Stage1 Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss1 {loss1.val:.3f} ({loss1.avg:.3f})\t'
                  'loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t'
                  'loss3 {loss3.val:.3f} ({loss3.avg:.3f})\t'
                  'acc3 {acc3.val:.3f} ({acc3.avg:.3f})\t'.format(
                epoch, step + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss1=loss1_meter, loss2=loss2_meter, loss3 = loss3_meter, acc3=accuracy3_meter))

    return losses.avg, loss1_meter.avg, loss2_meter.avg, loss3_meter.avg, accuracy1_meter.avg


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
            fs, fc, fv = model(input, viewt)
            loss = criterion(fv, viewt)  
            acc = accuracy(fv, viewt)  

        loss1_meter.update(loss.item(), bsz)
        accuracy1_meter.update(acc[0].item(), bsz)

    return loss1_meter.avg, accuracy1_meter.avg


'''
# self-realized adjust_learning_rate
def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.learning_rate * args.batch_size / 16
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
'''


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

        loss,loss1,loss2,loss3,acc1 = train(epoch, train_loader, model, criterion, criterion1, criterion2, criterion3, optimizer, args)
        time2 = time.time()

        loss1v, acc1v = validate(epoch, val_loader, model, criterion, criterion1, criterion2, criterion3, optimizer, args)
        print('val_loss1 {:.2f}, val_acc1 {:.2f}'.format(loss1v, acc1v))

        print('epoch {}, total time {:.2f}\t'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('stage1_val_loss1', loss1v, epoch) # evaluate
        # logger.log_value('stage1_val_acc1', acc1v, epoch) # 

        logger.log_value('stage1_loss', loss, epoch) 
        logger.log_value('stage1_loss1', loss1, epoch)  
        logger.log_value('stage1_loss2', loss2, epoch) 
        logger.log_value('stage1_loss3', loss3, epoch) 
        logger.log_value('stage1_acc3', acc1, epoch) 

        
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
