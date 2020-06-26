import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from mean_teacher import losses, ramps
from utils.util import FocalLoss, PULoss, SigmoidLoss, laplacian
from utils.metrics import ConfusionMatrix
from lenet_2conv_clf_oct_17_2018 import Lenet3D as Model
from adni_dataset import ADNI
from functions import *
from torchvision import transforms

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
parser.add_argument('--batch-size', '-b', type=int, default=64, help='batch-size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--modeldir', type=str, default="model/", help="Model path")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--loss', type=str, default='nnPU')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, help='workers')

# Self Paced 
parser.add_argument('--self-paced', type=boolean_string, default=True)
parser.add_argument('--self-paced-start', type=int, default=50)
parser.add_argument('--self-paced-stop', type=int, default = -1)
parser.add_argument('--self-paced-frequency', type=int, default=10)
parser.add_argument('--self-paced-type', type=str, default = "A")

parser.add_argument('--increasing', type=boolean_string, default=True)
parser.add_argument('--replacement', type=boolean_string, default=True)

parser.add_argument('--evaluation', action="store_true")
parser.add_argument('--top', type=float, default=0.5)
parser.add_argument('--soft-label', action="store_true")

parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--datapath', type=str, default="")

results = np.zeros(822)
switched = False
step = 0
args = None
single_epoch_steps = 0
def main():

    global args, switched, single_epoch_steps, step
    args = parser.parse_args()
    criterion = nn.CrossEntropyLoss().cuda()

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

    #step = args.ema_start * 2 + 1
    dataset_train_clean = ADNI("adni_dx_suvr_clean.csv", ids_train, [], '/ssd1/chenwy/adni', type="clean", transform = True)
    dataset_train_noisy = ADNI("adni_dx_suvr_clean.csv", ids_train, None, '/ssd1/chenwy/adni', type="noisy", transform = True)
    dataset_test = ADNI("adni_dx_suvr_clean.csv", ids_val, None, '/ssd1/chenwy/adni', type="clean", transform = False)
    dataloader_train_clean = DataLoader(dataset_train_clean, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    dataloader_train_noisy = DataLoader(dataset_train_noisy, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    
    model = create_model().cuda()

    params_list = [{'params': model.parameters(), 'lr': args.lr},] 
    #optimizer = torch.optim.Adam(params_list, lr=args.lr,
    #    weight_decay=args.weight_decay
    #) 
    optimizer = torch.optim.Adam(params_list, lr=args.lr,
        weight_decay=args.weight_decay
    )   
    stats_ = stats(args.modeldir, 0)
    #scheduler = torch.optim.lr_scheduler.(optimizer, args.epochs, eta_min = args.lr * 0.2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.6)
    if args.evaluation:
        print("Evaluation mode!")
    best_acc = 0
    val = []
    for epoch in range(args.epochs):

        trainPacc, trainNacc, trainPNacc = train(dataloader_train_clean, dataloader_train_noisy, model, criterion, optimizer, scheduler, epoch)
        valPacc, valNacc, valPNacc = validate(dataloader_test, model, criterion, epoch)

        #validate_2(dataloader_test, model, ema_model, criterion, consistency_criterion, epoch)
        stats_._update(trainPacc, trainNacc, trainPNacc, valPacc, valNacc, valPNacc)

        is_best = valPNacc > best_acc
        best_acc = max(valPNacc, best_acc)
        filename = []
        filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
        filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))

        dataset_train_noisy.shuffle()
        #dataloader_train_clean = DataLoader(dataset_train_clean, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
        dataloader_train_noisy = DataLoader(dataset_train_noisy, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    print(best_acc)

def train(clean_loader, noisy_loader, model, criterion, optimizer, scheduler, epoch):

    global step, switched, single_epoch_steps
    pacc = AverageMeter()
    nacc = AverageMeter()
    pnacc = AverageMeter()
    model.train()
    scheduler.step()
    print("Learning rate is {}".format(optimizer.param_groups[0]['lr']))
   
    for i, (X, left, right, Y, _, T, ids) in enumerate(noisy_loader):
        if args.gpu == None:
            X = X.cuda(args.gpu)
            left = left.cuda(args.gpu)
            right = right.cuda(args.gpu)
            Y = Y.cuda(args.gpu).long()
            T = T.cuda(args.gpu).long()
        else:
            X = X.cuda(args.gpu)
            left = left.cuda(args.gpu)
            right = right.cuda(args.gpu)
            Y = Y.cuda(args.gpu).long()
            T = T.cuda(args.gpu).long()

        output = model(X, left, right)
        smx = torch.sigmoid(output) # 计算sigmoid概率
        #print(smx)
        smx = torch.cat([1 - smx, smx], dim=1) # 组合成预测变量
        #if epoch >= args.self_paced_start: print(output)
        smxY = ((Y + 1) // 2).long() 
        loss = criterion(smx + 1e-10, smxY)
        predictions = torch.sign(output).long()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pacc_, nacc_, pnacc_, psize = accuracy(predictions, T)
        pacc.update(pacc_, psize)
        nacc.update(nacc_, X.size(0) - psize)
        pnacc.update(pnacc_, X.size(0))

    print('Noisy Epoch: [{0}]\t'
        'PACC {pacc.val:.3f} ({pacc.avg:.3f})\t'
        'NACC {nacc.val:.3f} ({nacc.avg:.3f})\t'
        'PNACC {pnacc.val:.3f} ({pnacc.avg:.3f})\t'.format(
        epoch, i, len(noisy_loader), pacc=pacc, nacc = nacc, pnacc=pnacc))

    
    return pacc.avg, nacc.avg, pnacc.avg

def validate(val_loader, model, criterion, epoch):
    pacc = AverageMeter()
    nacc = AverageMeter()
    pnacc = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (X, left, right, Y, _, T, ids) in enumerate(val_loader):
            # measure data loading time

            if args.gpu == None:
                X = X.cuda(args.gpu)
                left = left.cuda(args.gpu)
                right = right.cuda(args.gpu)
                Y = Y.cuda(args.gpu).float()
                T = T.cuda(args.gpu).long()
            else:
                X = X.cuda(args.gpu)
                left = left.cuda(args.gpu)
                right = right.cuda(args.gpu)
                Y = Y.cuda(args.gpu).float()
                T = T.cuda(args.gpu).long()
            # compute output
            output = model(X, left, right) 
            predictions = torch.sign(output).long()           
            pacc_, nacc_, pnacc_, psize = accuracy(predictions, T)
            pacc.update(pacc_, psize)
            nacc.update(nacc_, X.size(0) - psize)
            pnacc.update(pnacc_, X.size(0))
            

    print('Test [{0}]: \t'
                'PACC {pacc.val:.3f} ({pacc.avg:.3f})\t'
                'NACC {nacc.val:.3f} ({nacc.avg:.3f})\t'
                'PNACC {pnacc.val:.3f} ({pnacc.avg:.3f})\t'.format(
                epoch, pacc=pacc, nacc=nacc, pnacc=pnacc))
    print("=====================================")
    return pacc.avg, nacc.avg, pnacc.avg

def create_model(ema=False):
    model = Model()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


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
        #print(self.val)
        #print(n)
        #print('-----')
        self.sum += val * n
        self.count += n
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
    
    #print(pcorrect)
    #print(ncorrect)
    ptotal = torch.sum(target == 1).float()
    #print(ptotal)
    #print(ptotal)
    if ptotal == 0:
        return torch.tensor(0.).cuda(args.gpu), ncorrect / (batch_size - ptotal) * 100, correct / batch_size * 100, ptotal
    elif ptotal == batch_size:
        return pcorrect / ptotal * 100, torch.tensor(0.).cuda(args.gpu), correct / batch_size * 100, ptotal
    else:
        return pcorrect / ptotal * 100, ncorrect / (batch_size - ptotal) * 100, correct / batch_size * 100, ptotal


if __name__ == '__main__':
    main()