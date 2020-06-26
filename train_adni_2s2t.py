import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from mean_teacher import losses, ramps
from utils.util import FocalLoss, PULoss
from datasets import MNIST_Dataset, get_mnist, binarize_mnist_class
from cifar_datasets import CIFAR_Dataset, get_cifar, binarize_cifar_class
from adni_dataset import ADNI
from functions import *
from torchvision import transforms
from lenet_2conv_clf_oct_17_2018 import Lenet3D

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import shutil
import copy

from tensorboardX import SummaryWriter
from tqdm import tqdm

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch-size', '-b', type=int, default=512, help='batch-size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
)
parser.add_argument('--modeldir', type=str, default="model/", help="Model path")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--loss', type=str, default='nnPU')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, help='workers')

parser.add_argument('--weight', type=float, default=1.0)

# Self Paced 
parser.add_argument('--self-paced', type=boolean_string, default=True)
parser.add_argument('--self-paced-start', type=int, default=10)
parser.add_argument('--self-paced-stop', type=int, default=50)
parser.add_argument('--self-paced-frequency', type=int, default=10)
parser.add_argument('--self-paced-type', type=str, default = "A")

parser.add_argument('--increasing', type=boolean_string, default=True)
parser.add_argument('--replacement', type=boolean_string, default=True)

parser.add_argument('--mean-teacher', type=boolean_string, default=True)
parser.add_argument('--ema-start', type=int, default=50)
parser.add_argument('--ema-decay', type=float, default=0.999)
parser.add_argument('--consistency', type=float, default = 0.3)
parser.add_argument('--consistency-rampup', type=int, default = 400)

parser.add_argument('--top1', type=float, default=0.4)
parser.add_argument('--top2', type=float, default=0.6)
parser.add_argument('--soft-label', action="store_true")
parser.add_argument('--type', type=str, default = "mu")
parser.add_argument('--alpha', default = 0.1, type = float)
step = 0
results = np.zeros(61000)
switched = False
args = None
results1 = None
results2 = None

def main():

    global args, switched
    args = parser.parse_args()
    criterion = get_criterion()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    ids_train = np.load("rid.image_id.train.adni.npy")
    ids_val = np.load("rid.image_id.test.adni.npy")
    # load metadata from csv ######################################
    df = pd.read_csv("adni_dx_suvr_clean.csv")
    df = df.fillna('')
    tmp = []
    for i in range(len(ids_train)):
        id = ids_train[i]
        if '.' in id:
            id = id.split('.')
            dx = df[(df['RID'] == int(id[0])) & (df['MRI ImageID'] == int(id[1]))]['DX'].values[0]
        else:
            dx = df[(df['RID'] == int(id)) & (df['MRI ImageID'] == "")]['DX'].values[0]
        # train on AD/MCI/NL ([1,2,3]) or only AD/NL ([1,3])
        if dx in [1, 3]: tmp.append(ids_train[i])
    ids_train = np.array(tmp)
    tmp = []
    for i in range(len(ids_val)):
        id = ids_val[i]
        if '.' in id:
            id = id.split('.')
            dx = df[(df['RID'] == int(id[0])) & (df['MRI ImageID'] == int(id[1]))]['DX'].values[0]
        else:
            dx = df[(df['RID'] == int(id)) & (df['MRI ImageID'] == "")]['DX'].values[0]
        # train on AD/MCI/NL ([1,2,3]) or only AD/NL ([1,3])
        if dx in [1, 3]: tmp.append(ids_val[i])
    ids_val = np.array(tmp)
    print(len(ids_train), len(ids_val))


    dataset_train1_clean = ADNI("adni_dx_suvr_clean.csv", ids_train, [], '/ssd1/chenwy/adni', type="clean", transform = True)
    dataset_train2_clean = ADNI("adni_dx_suvr_clean.csv", ids_train, [], '/ssd1/chenwy/adni', type="clean", transform = True)
    dataset_train1_noisy = ADNI("adni_dx_suvr_clean.csv", ids_train, None, '/ssd1/chenwy/adni', type="noisy", transform = True)
    dataset_train2_noisy = ADNI("adni_dx_suvr_clean.csv", ids_train, None, '/ssd1/chenwy/adni', type="noisy", transform = True)
    dataset_test = ADNI("adni_dx_suvr_clean.csv", ids_val, None, '/ssd1/chenwy/adni', type="clean", transform = False)
    criterion.update_p(0.43)

        
    dataloader_train1_clean = None
    #dataloader_train1_clean = DataLoader(dataset_train1_clean, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    dataloader_train1_noisy = DataLoader(dataset_train1_noisy, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    dataloader_train2_clean = None
    #dataloader_train2_clean = DataLoader(dataset_train2_clean, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    dataloader_train2_noisy = DataLoader(dataset_train2_noisy, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    consistency_criterion = losses.softmax_mse_loss
    
    model1 = create_lenet_model()
    model2 = create_lenet_model()
    ema_model1 = create_lenet_model(ema = True)
    ema_model2 = create_lenet_model(ema = True)
    if args.gpu is not None:
        model1 = model1.cuda(args.gpu)
        model2 = model2.cuda(args.gpu)
        ema_model1 = ema_model1.cuda(args.gpu)
        ema_model2 = ema_model2.cuda(args.gpu)
    else:
        model1 = model1.cuda(args.gpu)
        model2 = model2.cuda(args.gpu)
        ema_model1 = ema_model1.cuda(args.gpu)
        ema_model2 = ema_model2.cuda(args.gpu)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr,
        weight_decay=args.weight_decay
    )   
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr,
        weight_decay=args.weight_decay
    )  

    stats_ = stats(args.modeldir, 0)
    #scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[15, 60], gamma=0.7)
    #scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[15, 60], gamma=0.7)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, args.epochs)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, args.epochs)

    best_acc1 = 0
    best_acc2 = 0
    best_acc3 = 0
    best_acc4 = 0
    for epoch in range(args.epochs):
        print("Self paced status: {}".format(check_self_paced(epoch)))
        print("Mean Teacher status: {}".format(check_mean_teacher(epoch)))
        if check_mean_teacher(epoch) and not check_mean_teacher(epoch - 1) and not switched:
            ema_model1.load_state_dict(model1.state_dict())
            ema_model2.load_state_dict(model2.state_dict())
            switched = True
            print("SWITCHED!")

        trainPacc, trainNacc, trainPNacc = train(dataloader_train1_clean, dataloader_train1_noisy, dataloader_train2_clean, dataloader_train2_noisy, model1, model2, ema_model1, ema_model2, criterion, consistency_criterion, optimizer1, scheduler1, optimizer2, scheduler2, epoch)

        valPacc, valNacc, valPNacc1, valPNacc2, valPNacc3, valPNacc4 = validate(dataloader_test, model1, model2, ema_model1, ema_model2, epoch)
        #print(valPacc, valNacc, valPNacc1, valPNacc2, valPNacc3)
        stats_._update(trainPacc, trainNacc, trainPNacc, valPacc, valNacc, valPNacc1)

        is_best1 = valPNacc1 > best_acc1
        is_best2 = valPNacc2 > best_acc2
        is_best3 = valPNacc3 > best_acc3
        is_best4 = valPNacc4 > best_acc4
        best_acc1 = max(valPNacc1, best_acc1)
        best_acc2 = max(valPNacc2, best_acc2)
        best_acc3 = max(valPNacc3, best_acc3)
        best_acc4 = max(valPNacc4, best_acc4)
        filename = []
        filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
        filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))

        if (check_self_paced(epoch)) and (epoch - args.self_paced_start) % args.self_paced_frequency == 0:

            dataloader_train1_clean, dataloader_train1_noisy, dataloader_train2_clean, dataloader_train2_noisy = update_dataset(model1, model2, ema_model1, ema_model2, dataset_train1_clean, dataset_train1_noisy, dataset_train2_clean, dataset_train2_noisy, epoch)

        plot_curve(stats_, args.modeldir, 'model', True)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model1.state_dict(),
            'best_prec1': best_acc1,
        }, is_best1, filename)

        dataset_train1_noisy.shuffle()
        dataset_train2_noisy.shuffle()
        dataloader_train1_noisy = DataLoader(dataset_train1_noisy, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
        dataloader_train2_noisy = DataLoader(dataset_train2_noisy, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    print(best_acc1)
    print(best_acc2)
    print(best_acc3)
    print(best_acc4)
def train(clean1_loader, noisy1_loader, clean2_loader, noisy2_loader, model1, model2, ema_model1, ema_model2, criterion, consistency_criterion, optimizer1, scheduler1, optimizer2, scheduler2, epoch, warmup = False, self_paced_pick = 0):

    global step, switched
    batch_time = AverageMeter()
    data_time = AverageMeter()

    pacc1 = AverageMeter()
    nacc1 = AverageMeter()
    pnacc1 = AverageMeter()
    pacc2 = AverageMeter()
    nacc2 = AverageMeter()
    pnacc2 = AverageMeter()
    pacc3 = AverageMeter()
    nacc3 = AverageMeter()
    pnacc3 = AverageMeter()
    pacc4 = AverageMeter()
    nacc4 = AverageMeter()
    pnacc4 = AverageMeter()
    model1.train()
    model2.train()
    ema_model1.train()
    ema_model2.train()
    end = time.time()
    consistency_weight = get_current_consistency_weight(epoch - 30)
    if not warmup: 
        scheduler1.step()
        scheduler2.step()
    resultt = np.zeros(61000)
    
    if clean1_loader:
        for i, (X, left, right, _, Y, T, ids) in enumerate(clean1_loader):
            # measure data loading time
            
            data_time.update(time.time() - end)
            X = X.cuda(args.gpu)
            left = left.cuda(args.gpu)
            right = right.cuda(args.gpu)
            Y = Y.cuda(args.gpu).float()
            T = T.cuda(args.gpu).long()
            # compute output
            output1 = model1(X, left, right)
            output2 = model2(X, left, right)
            with torch.no_grad():
                ema_output1 = ema_model1(X, left, right)

            consistency_loss = consistency_weight * \
            consistency_criterion(output1, ema_output1) / X.shape[0]

            predictiont1   = torch.sign(ema_output1).long()
            predictions1 = torch.sign(output1).long() # 否则使用自己的结果

            smx1 = torch.sigmoid(output1) # 计算sigmoid概率
            smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量
            smxY = ((Y + 1) // 2).long() # 分类结果，0-1分类
            smx2 = torch.sigmoid(output2) # 计算sigmoid概率
            smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量

            if args.soft_label:
                aux1 = - torch.sum(smx1 * torch.log(smx1 + 1e-10)) / smx1.shape[0]
                aux2 = - torch.sum(smx2 * torch.log(smx2 + 1e-10)) / smx2.shape[0]
            else:
                smxY = smxY.float()
                smxY = smxY.view(-1, 1)
                smxY = torch.cat([1 - smxY, smxY], dim = 1)
                aux1 = - torch.sum(smxY * torch.log(smx1 + 1e-10)) / smxY.shape[0] # 计算Xent loss
                aux2 = - torch.sum(smxY * torch.log(smx2 + 1e-10)) / smxY.shape[0] # 计算Xent loss

            loss = aux1


            if check_mean_teacher(epoch):
                loss += consistency_loss

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            
            pacc_1, nacc_1, pnacc_1, psize = accuracy(predictions1, T) # 使用T来计算预测准确率
            pacc_3, nacc_3, pnacc_3, psize = accuracy(predictiont1, T) 
            pacc1.update(pacc_1, psize)
            nacc1.update(nacc_1, X.size(0) - psize)
            pnacc1.update(pnacc_1, X.size(0))
            pacc3.update(pacc_3, psize)
            nacc3.update(nacc_3, X.size(0) - psize)
            pnacc3.update(pnacc_3, X.size(0))
    
    if clean2_loader:
        for i, (X, left, right,  _, Y, T, ids) in enumerate(clean2_loader):
            # measure data loading time
            
            data_time.update(time.time() - end)
            X = X.cuda(args.gpu)
            left = left.cuda(args.gpu)
            right = right.cuda(args.gpu)
            Y = Y.cuda(args.gpu).float()
            T = T.cuda(args.gpu).long()
            # compute output
            output1 = model1(X, left, right)
            output2 = model2(X, left, right)
            with torch.no_grad():
                ema_output2 = ema_model2(X, left, right)

            consistency_loss = consistency_weight * \
            consistency_criterion(output2, ema_output2) / X.shape[0]

            predictiont2 = torch.sign(ema_output2).long()
            predictions2 = torch.sign(output2).long()

            smx1 = torch.sigmoid(output1) # 计算sigmoid概率
            smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量

            smxY = ((Y + 1) // 2).long() # 分类结果，0-1分类

            smx2 = torch.sigmoid(output2) # 计算sigmoid概率
            smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量

            if args.soft_label:
                aux1 = - torch.sum(smx1 * torch.log(smx1 + 1e-10)) / smx1.shape[0]
                aux2 = - torch.sum(smx2 * torch.log(smx2 + 1e-10)) / smx2.shape[0]
            else:
                smxY = smxY.float()
                smxY = smxY.view(-1, 1)
                smxY = torch.cat([1 - smxY, smxY], dim = 1)
                aux1 = - torch.sum(smxY * torch.log(smx1 + 1e-10)) / smxY.shape[0] # 计算Xent loss
                aux2 = - torch.sum(smxY * torch.log(smx2 + 1e-10)) / smxY.shape[0] # 计算Xent loss

            loss = aux2

            if check_mean_teacher(epoch):
                loss += consistency_loss
                
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

            pacc_2, nacc_2, pnacc_2, psize = accuracy(predictions2, T)
            pacc_4, nacc_4, pnacc_4, psize = accuracy(predictiont2, T) 
            pacc2.update(pacc_2, psize)
            nacc2.update(nacc_2, X.size(0) - psize)
            pnacc2.update(pnacc_2, X.size(0))
            pacc4.update(pacc_4, psize)
            nacc4.update(nacc_4, X.size(0) - psize)
            pnacc4.update(pnacc_4, X.size(0))

        if check_mean_teacher(epoch):
            update_ema_variables(model1, ema_model1, args.ema_decay, step) # 更新ema参数
            update_ema_variables(model2, ema_model2, args.ema_decay, step)
            step += 1

        print('Epoch Clean : [{0}]\t'
                'PACC1 {pacc1.val:.3f} ({pacc1.avg:.3f})\t'
                'NACC1 {nacc1.val:.3f} ({nacc1.avg:.3f})\t'
                'PNACC1 {pnacc1.val:.3f} ({pnacc1.avg:.3f})\t'
                'PACC2 {pacc2.val:.3f} ({pacc2.avg:.3f})\t'
                'NACC2 {nacc2.val:.3f} ({nacc2.avg:.3f})\t'
                'PNACC2 {pnacc2.val:.3f} ({pnacc2.avg:.3f})\t'
                'PACC3 {pacc3.val:.3f} ({pacc3.avg:.3f})\t'
                'NACC3 {nacc3.val:.3f} ({nacc3.avg:.3f})\t'
                'PNACC3 {pnacc3.val:.3f} ({pnacc3.avg:.3f})\t'
                'PACC4 {pacc4.val:.3f} ({pacc4.avg:.3f})\t'
                'NACC4 {nacc4.val:.3f} ({nacc4.avg:.3f})\t'
                'PNACC4 {pnacc4.val:.3f} ({pnacc4.avg:.3f})\t'.format(
                epoch, pacc1=pacc1, nacc1=nacc1, pnacc1=pnacc1, 
                pacc2=pacc2, nacc2=nacc2, pnacc2=pnacc2, pacc3=pacc3, nacc3=nacc3, pnacc3=pnacc3, 
                pacc4=pacc4, nacc4=nacc4, pnacc4=pnacc4))
    
    #if epoch > args.self_paced_start: criterion.update_p(0.05)
    if (args.dataset == 'cifar'):
        criterion.update_p((20000 - self_paced_pick / 2) / (50000 - self_paced_pick))
        print("Setting Pi_P to {}".format((20000 - self_paced_pick / 2) / (50000 - self_paced_pick)))

    for i, (X, left, right, Y, _, T, ids) in enumerate(noisy1_loader):
        X = X.cuda(args.gpu)
        left = left.cuda(args.gpu)
        right = right.cuda(args.gpu)
        Y = Y.cuda(args.gpu).float()
        T = T.cuda(args.gpu).long()

        # compute output
        output1 = model1(X, left, right)
        output2 = model2(X, left, right)

        with torch.no_grad():
            ema_output1 = ema_model1(X, left, right)
        #if epoch >= args.self_paced_start: criterion.update_p(0.5)
        _, loss = criterion(output1, Y)
        consistency_loss = consistency_weight * \
        consistency_criterion(output1, ema_output1) / X.shape[0]

        predictions1 = torch.sign(output1).long()
        predictiont1 = torch.sign(ema_output1).long()

        smx1 = torch.sigmoid(output1) # 计算sigmoid概率
        smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量

        smxY = ((Y + 1) // 2).long() # 分类结果，0-1分类

        smx2 = torch.sigmoid(output2) # 计算sigmoid概率
        smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量

        if args.type == 'mu' and check_mean_teacher(epoch):
            aux = F.mse_loss(smx1[:, 0], smx2[:, 0].detach_())
            if aux < loss * args.alpha:
                loss += aux

        if check_mean_teacher(epoch):
            loss += consistency_loss
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        pacc_3, nacc_3, pnacc_3, psize = accuracy(predictiont1, T) 
        pacc_1, nacc_1, pnacc_1, psize = accuracy(predictions1, T) # 使用T来计算预测准确率
        pacc1.update(pacc_1, psize)
        nacc1.update(nacc_1, X.size(0) - psize)
        pnacc1.update(pnacc_1, X.size(0))
        pacc3.update(pacc_3, psize)
        nacc3.update(nacc_3, X.size(0) - psize)
        pnacc3.update(pnacc_3, X.size(0))

    for i, (X, left, right, Y, _, T, ids) in enumerate(noisy2_loader):

        X = X.cuda(args.gpu)
        left = left.cuda(args.gpu)
        right = right.cuda(args.gpu)
        Y = Y.cuda(args.gpu).float()
        T = T.cuda(args.gpu).long()

        # compute output
        output1 = model1(X, left, right)
        output2 = model2(X, left, right)
        with torch.no_grad():
            ema_output2 = ema_model2(X, left, right)

        _, loss = criterion(output2, Y)
        consistency_loss = consistency_weight * \
        consistency_criterion(output2, ema_output2) / X.shape[0]
        #print(loss2)
        predictions2 = torch.sign(output2).long()
        predictiont2 = torch.sign(ema_output2).long()

        smx1 = torch.sigmoid(output1) # 计算sigmoid概率
        smx1 = torch.cat([1 - smx1, smx1], dim=1) # 组合成预测变量

        smxY = ((Y + 1) // 2).long() # 分类结果，0-1分类

        smx2 = torch.sigmoid(output2) # 计算sigmoid概率
        smx2 = torch.cat([1 - smx2, smx2], dim=1) # 组合成预测变量
        
        if args.type == 'mu' and check_mean_teacher(epoch):
            aux = F.mse_loss(smx2[:, 0], smx1[:, 0].detach_())
            if aux < loss * args.alpha:
                loss += aux

        if check_mean_teacher(epoch):
            loss += consistency_loss

        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        pacc_2, nacc_2, pnacc_2, psize = accuracy(predictions2, T)
        pacc_4, nacc_4, pnacc_4, psize = accuracy(predictiont2, T) 
        pacc2.update(pacc_2, psize)
        nacc2.update(nacc_2, X.size(0) - psize)
        pnacc2.update(pnacc_2, X.size(0))
        pacc4.update(pacc_4, psize)
        nacc4.update(nacc_4, X.size(0) - psize)
        pnacc4.update(pnacc_4, X.size(0))

    if check_mean_teacher(epoch):
        update_ema_variables(model1, ema_model1, args.ema_decay, step) # 更新ema参数
        update_ema_variables(model2, ema_model2, args.ema_decay, step)
        step += 1

    print('Epoch Noisy : [{0}]\t'
            'PACC1 {pacc1.val:.3f} ({pacc1.avg:.3f})\t'
            'NACC1 {nacc1.val:.3f} ({nacc1.avg:.3f})\t'
            'PNACC1 {pnacc1.val:.3f} ({pnacc1.avg:.3f})\t'
            'PACC2 {pacc2.val:.3f} ({pacc2.avg:.3f})\t'
            'NACC2 {nacc2.val:.3f} ({nacc2.avg:.3f})\t'
            'PNACC2 {pnacc2.val:.3f} ({pnacc2.avg:.3f})\t'
            'PACC3 {pacc3.val:.3f} ({pacc3.avg:.3f})\t'
            'NACC3 {nacc3.val:.3f} ({nacc3.avg:.3f})\t'
            'PNACC3 {pnacc3.val:.3f} ({pnacc3.avg:.3f})\t'
            'PACC4 {pacc4.val:.3f} ({pacc4.avg:.3f})\t'
            'NACC4 {nacc4.val:.3f} ({nacc4.avg:.3f})\t'
            'PNACC4 {pnacc4.val:.3f} ({pnacc4.avg:.3f})\t'.format(
            epoch, pacc1=pacc1, nacc1=nacc1, pnacc1=pnacc1, 
            pacc2=pacc2, nacc2=nacc2, pnacc2=pnacc2, pacc3=pacc3, nacc3=nacc3, pnacc3=pnacc3, 
            pacc4=pacc4, nacc4=nacc4, pnacc4=pnacc4))

    
    return pacc1.avg, nacc1.avg, pnacc1.avg


def validate(val_loader, model1, model2, ema_model1, ema_model2, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pacc = AverageMeter()
    nacc = AverageMeter()
    pnacc1 = AverageMeter()
    pnacc2 = AverageMeter()
    pnacc3 = AverageMeter()
    pnacc4 = AverageMeter()
    model1.eval()
    model2.eval()
    ema_model1.eval()
    ema_model2.eval()
    end = time.time()
    
    with torch.no_grad():
        for i, (X, left, right, Y, _, T, ids) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            X = X.cuda(args.gpu)
            left = left.cuda(args.gpu)
            right = right.cuda(args.gpu)
            Y = Y.cuda(args.gpu).float()
            T = T.cuda(args.gpu).long()

            # compute output
            output1 = model1(X, left, right)
            output2 = model2(X, left, right)
            ema_output1 = ema_model1(X, left, right)    
            ema_output2 = ema_model2(X, left, right)    
            predictions1 = torch.sign(output1).long()
            predictions2 = torch.sign(output2).long()
            predictiont1 = torch.sign(ema_output1).long()
            predictiont2 = torch.sign(ema_output2).long()
            
            pacc_, nacc_, pnacc_, psize = accuracy(predictions1, T)
            pacc.update(pacc_, X.size(0))
            nacc.update(nacc_, X.size(0))
            pnacc1.update(pnacc_, X.size(0))
            pacc_, nacc_, pnacc_, psize = accuracy(predictions2, T)
            pnacc2.update(pnacc_, X.size(0))
            pacc_, nacc_, pnacc_, psize = accuracy(predictiont1, T)
            pnacc3.update(pnacc_, X.size(0))
            pacc_, nacc_, pnacc_, psize = accuracy(predictiont2, T)
            pnacc4.update(pnacc_, X.size(0))


    print('Test [{0}]: \t'
                'PNACC1 {pnacc1.val:.3f} ({pnacc1.avg:.3f})\t'
                'PNACC2 {pnacc2.val:.3f} ({pnacc2.avg:.3f})\t'
                'PNACC3 {pnacc3.val:.3f} ({pnacc3.avg:.3f})\t'
                'PNACC4 {pnacc4.val:.3f} ({pnacc4.avg:.3f})\t'.format(
                epoch, pnacc1=pnacc1, pnacc2=pnacc2, pnacc3=pnacc3, pnacc4 = pnacc4))
    print("=====================================")
    return pacc.avg, nacc.avg, pnacc1.avg, pnacc2.avg, pnacc3.avg , pnacc4.avg

def create_lenet_model(ema=False):
    model = Lenet3D()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def update_ema_variables(model, ema_model, alpha, global_step):

    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, (param))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_criterion():
    weights = [float(args.weight), 1.0]
    class_weights = torch.FloatTensor(weights)

    class_weights = class_weights.cuda(args.gpu)
    if args.loss == 'Xent':
        criterion = PULoss(Probability_P=0.49, loss_fn="Xent")
    elif args.loss == 'nnPU':
        criterion = PULoss(Probability_P=0.49)
    elif args.loss == 'Focal':
        class_weights = torch.FloatTensor(weights).cuda(args.gpu)
        criterion = FocalLoss(gamma=0, weight=class_weights, one_hot=False)
    elif args.loss == 'uPU':
        criterion = PULoss(Probability_P=0.49, nnPU=False)
    elif args.loss == 'Xent_weighted':
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    return criterion

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        #print(val, n)
        if self.count == 0:
            self.avg = 0
        else:
            self.avg = self.sum / self.count

def accuracy(output, target):
    with torch.no_grad():
        
        batch_size = float(target.size(0))
        
        output = output.view(-1)
        correct = torch.sum(output == target).float()
        
        pcorrect = torch.sum(output[target==1] == target[target == 1]).float()
        ncorrect = correct - pcorrect
    
    ptotal = torch.sum(target == 1).float()

    if ptotal == 0:
        return torch.tensor(0.).cuda(args.gpu), ncorrect / (batch_size - ptotal) * 100, correct / batch_size * 100, ptotal
    elif ptotal == batch_size:
        return pcorrect / ptotal * 100, torch.tensor(0.).cuda(args.gpu), correct / batch_size * 100, ptotal
    else:
        return pcorrect / ptotal * 100, ncorrect / (batch_size - ptotal) * 100, correct / batch_size * 100, ptotal

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])

def update_dataset(model1, model2, ema_model1, ema_model2, dataset_train1_clean, dataset_train1_noisy, dataset_train2_clean, dataset_train2_noisy, epoch, ratio=0.5):
    #global results
    global results1, results2
    dataset_train1_noisy.reset_ids()
    dataset_train1_noisy.set_type("clean")
    dataset_train2_noisy.reset_ids()
    dataset_train2_noisy.set_type("clean")
    dataloader_train1 = DataLoader(dataset_train1_noisy, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)
    dataloader_train2 = DataLoader(dataset_train2_noisy, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)
    if args.dataset == 'mnist':
        results1 = np.zeros(61000) # rid.imageid: p_pos # 存储概率结果
        results2 = np.zeros(61000)
    elif args.dataset == 'cifar':
        results1 = np.zeros(51000)
        results2 = np.zeros(51000)
    elif args.dataset == 'ADNI':
        results1 = np.zeros(822)
        results2 = np.zeros(822)
    model1.eval()
    model2.eval()
    # validation #######################
    with torch.no_grad():
        for i_batch, (X, left, right, _, _, T, ids) in enumerate(tqdm(dataloader_train1)):
            #images, lefts, rights, ages, genders, edus, apoes, labels, pu_labels, ids = Variable(sample_batched['mri']).cuda(args.gpu), Variable(sample_batched['left']).cuda(args.gpu), Variable(sample_batched['right']).cuda(args.gpu), Variable(sample_batched['age']).cuda(args.gpu), Variable(sample_batched['gender']).cuda(args.gpu), Variable(sample_batched['edu']).cuda(args.gpu), Variable(sample_batched['apoe']).cuda(args.gpu), Variable(sample_batched['label']).view(-1).type(torch.LongTensor).cuda(args.gpu), Variable(sample_batched['pu_label']).view(-1).type(torch.LongTensor).cuda(args.gpu), sample_batched['id']
            X = X.cuda(args.gpu)
            left = left.cuda(args.gpu)
            right = right.cuda(args.gpu)
            #Y = Y.cuda(args.gpu)
            #Y = Y.float()
            # ===================forward====================
            if check_mean_teacher(epoch):
                output1 = ema_model1(X, left, right)
            else:
                output1 = model1(X, left, right)
            prob1 = torch.sigmoid(output1).view(-1).cpu().numpy()       
            results1[ids.view(-1).numpy()] = prob1

        for i_batch, (X, left, right, _, _, T, ids) in enumerate(tqdm(dataloader_train2)):
            #images, lefts, rights, ages, genders, edus, apoes, labels, pu_labels, ids = Variable(sample_batched['mri']).cuda(args.gpu), Variable(sample_batched['left']).cuda(args.gpu), Variable(sample_batched['right']).cuda(args.gpu), Variable(sample_batched['age']).cuda(args.gpu), Variable(sample_batched['gender']).cuda(args.gpu), Variable(sample_batched['edu']).cuda(args.gpu), Variable(sample_batched['apoe']).cuda(args.gpu), Variable(sample_batched['label']).view(-1).type(torch.LongTensor).cuda(args.gpu), Variable(sample_batched['pu_label']).view(-1).type(torch.LongTensor).cuda(args.gpu), sample_batched['id']
            X = X.cuda(args.gpu)
            left = left.cuda(args.gpu)
            right = right.cuda(args.gpu)
            #Y = Y.cuda(args.gpu)
            #Y = Y.float()
            # ===================forward====================
            if check_mean_teacher(epoch):
                output2 = ema_model2(X, left, right)
            else:
                output2 = model2(X, left, right)
            prob2 = torch.sigmoid(output2).view(-1).cpu().numpy()       
            results2[ids.view(-1).numpy()] = prob2

    # adni_dataset_train.update_labels(results, ratio)
    # dataset_origin = dataset_train
    ids_noisy1 = dataset_train1_clean.update_ids(results1, epoch, ratio = ratio) # 返回的是noisy ids
    ids_noisy2 = dataset_train2_clean.update_ids(results2, epoch, ratio = ratio)
    dataset_train1_noisy.set_ids(ids_noisy1) # 将noisy ids更新进去
    dataset_train1_noisy.set_type("noisy")
    dataset_train2_noisy.set_ids(ids_noisy2) # 将noisy ids更新进去
    dataset_train2_noisy.set_type("noisy")

    #assert np.all(dataset_train_noisy.ids == ids_noisy) # 确定更新了
    #dataloader_origin = DataLoader(dataset_origin, batch_size=args.batch_size, num_workers=4, drop_last=True, shuffle=True, pin_memory=True)
    dataloader_train1_clean = DataLoader(dataset_train1_clean, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    dataloader_train1_noisy = DataLoader(dataset_train1_noisy, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)
    dataloader_train2_clean = DataLoader(dataset_train2_clean, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    dataloader_train2_noisy = DataLoader(dataset_train2_noisy, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)
    return dataloader_train1_clean, dataloader_train1_noisy, dataloader_train2_clean, dataloader_train2_noisy

def check_mean_teacher(epoch):
    if not args.mean_teacher:
        return False
    elif epoch < args.ema_start:
        return False
    else:
        return True

def check_self_paced(epoch):
    if not args.self_paced:
        return False
    elif args.self_paced and epoch >= args.self_paced_stop:
        return False
    elif args.self_paced and epoch < args.self_paced_start:
        return False
    else: return True



if __name__ == '__main__':
    main()