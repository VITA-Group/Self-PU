import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from mean_teacher import losses, ramps
from utils.util import FocalLoss, PULoss, SigmoidLoss, laplacian
from utils.metrics import ConfusionMatrix
from models import MultiLayerPerceptron as Model
from models import CNN
from datasets import MNIST_Dataset, get_mnist, binarize_mnist_class
from cifar_datasets import CIFAR_Dataset, get_cifar, binarize_cifar_class
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
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch-size', '-b', type=int, default=256, help='batch-size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--log-path', type=str, default='logs/', help='Log path')
parser.add_argument('--modeldir', type=str, default="model/", help="Model path")
parser.add_argument('--task-name', type=str, default="temp")
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--loss', type=str, default='nnPU')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, help='workers')

# Mean Teacher
parser.add_argument('--mean-teacher', type=boolean_string, default=True)
parser.add_argument('--ema-start', type=int, default=50)
parser.add_argument('--ema-decay', type=float, default=0.999)
parser.add_argument('--consistency', type=float, default = 0.3)
parser.add_argument('--consistency-rampup', type=int, default = 400)

parser.add_argument('--weight', type=float, default=1.0)

# Self Paced 
parser.add_argument('--self-paced', type=boolean_string, default=True)
parser.add_argument('--self-paced-start', type=int, default=10)
parser.add_argument('--self-paced-stop', type=int, default = 50)
parser.add_argument('--self-paced-frequency', type=int, default=10)
parser.add_argument('--self-paced-type', type=str, default = "A")

parser.add_argument('--increasing', type=boolean_string, default=True)
parser.add_argument('--replacement', type=boolean_string, default=True)

parser.add_argument('--evaluation', action="store_true")
parser.add_argument('--top', type=float, default=0.5)
parser.add_argument('--soft-label', action="store_true")

parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--datapath', type=str, default="")


results = np.zeros(61000)
switched = False
step = 0
single_epoch_steps = 0
args = None
def main():

    global args, switched, single_epoch_steps, step
    args = parser.parse_args()

    criterion = get_criterion()

    torch.cuda.set_device(int(args.gpu))
    cudnn.benchmark = True

    if args.dataset == "mnist":
        (trainX, trainY), (testX, testY) = get_mnist()
        _trainY, _testY = binarize_mnist_class(trainY, testY)


        dataset_train_clean = MNIST_Dataset(1000, 60000, 
            trainX, _trainY, testX, _testY, split='train', ids=[],
            increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top = args.top, type="clean", seed = args.seed)
        # clean dataset初始化为空
        dataset_train_noisy = MNIST_Dataset(1000, 60000, 
            trainX, _trainY, testX, _testY, split='train',
            increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top = args.top, type="noisy", seed = args.seed)

        dataset_train_noisy.copy(dataset_train_clean) # 和clean dataset使用相同的随机顺序
        dataset_train_noisy.reset_ids() # 让初始化的noisy dataset使用全部数据

        dataset_test = MNIST_Dataset(1000, 60000, 
            trainX, _trainY, testX, _testY, split='test',
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top = args.top, type="clean", seed = args.seed)
    elif args.dataset == 'cifar':
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        } 
        (trainX, trainY), (testX, testY) = get_cifar()
        _trainY, _testY = binarize_cifar_class(trainY, testY)
        dataset_train_clean = CIFAR_Dataset(1000, 50000, 
            trainX, _trainY, testX, _testY, split='train', ids=[],
            increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top = args.top, transform = data_transforms['train'], type="clean", seed = args.seed)
        # clean dataset初始化为空
        dataset_train_noisy = CIFAR_Dataset(1000, 50000, 
            trainX, _trainY, testX, _testY, split='train',
            increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top = args.top, transform = data_transforms['train'], type="noisy", seed = args.seed)

        dataset_train_noisy.copy(dataset_train_clean) # 和clean dataset使用相同的随机顺序
        dataset_train_noisy.reset_ids() # 让初始化的noisy dataset使用全部数据

        dataset_test = CIFAR_Dataset(1000, 50000, 
            trainX, _trainY, testX, _testY, split='test',
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top = args.top, transform = data_transforms['val'], type="clean", seed = args.seed)

        criterion.update_p(0.4)

    assert np.all(dataset_train_noisy.X == dataset_train_clean.X)
    assert np.all(dataset_train_noisy.Y == dataset_train_clean.Y)
    assert np.all(dataset_train_noisy.oids == dataset_train_clean.oids)
    assert np.all(dataset_train_noisy.T == dataset_train_clean.T)

    #step = args.ema_start * 2 + 1

    if len(dataset_train_clean) > 0:
        dataloader_train_clean = DataLoader(dataset_train_clean, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    else:
        dataloader_train_clean = None
    
    if len(dataset_train_noisy) > 0:
        dataloader_train_noisy = DataLoader(dataset_train_noisy, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    else:
        dataloader_train_noisy = None
    
    if len(dataset_test):
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True)
    else:
        dataloader_test = None
    

    single_epoch_steps = len(dataloader_train_noisy) + 1
    print('Steps: {}'.format(single_epoch_steps))
    consistency_criterion = losses.softmax_mse_loss
    if args.dataset == 'mnist':
        model = create_model()
        ema_model = create_model(ema = True)
    elif args.dataset == 'cifar':
        model = create_cifar_model()
        ema_model = create_cifar_model(ema = True)

    if args.gpu is not None:
        model = model.cuda()
        ema_model = ema_model.cuda()
    else:
        model = model.cuda()
        ema_model = ema_model.cuda()

    params_list = [{'params': model.parameters(), 'lr': args.lr},] 
    #optimizer = torch.optim.Adam(params_list, lr=args.lr,
    #    weight_decay=args.weight_decay
    #) 
    optimizer = torch.optim.Adam(params_list, lr=args.lr,
        weight_decay=args.weight_decay
    )   
    stats_ = stats(args.modeldir, 0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = args.lr * 0.2)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.6)
    if args.evaluation:
        print("Evaluation mode!")
    best_acc = 0
    for epoch in range(args.warmup):
        print("Warming up {}/{}".format(epoch + 1, args.warmup))

        trainPacc, trainNacc, trainPNacc = train(dataloader_train_clean, dataloader_train_noisy, model, ema_model, criterion, consistency_criterion, optimizer, scheduler, -1, warmup = True)

        valPacc, valNacc, valPNacc = validate(dataloader_test, model, ema_model, criterion, consistency_criterion, -1)

        dataset_train_noisy.shuffle()
        dataloader_train_noisy = DataLoader(dataset_train_noisy, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    val = []
    for epoch in range(args.epochs):
        print("Self paced status: {}".format(check_self_paced(epoch)))
        print("Mean teacher status: {}".format(check_mean_teacher(epoch)))
        print("Noisy status: {}".format(check_noisy(epoch)))

        if check_mean_teacher(epoch) and (not check_mean_teacher(epoch - 1)) and not switched:
            model.eval()
            ema_model.eval()
            ema_model.load_state_dict(model.state_dict())
            switched = True
            print("SWITCHED!")
            validate(dataloader_test, model, ema_model, criterion, consistency_criterion, epoch)
            validate(dataloader_test, ema_model, model, criterion, consistency_criterion, epoch)
            model.train()
            ema_model.train()
        if epoch == 0:
            switched = False

        if (not check_mean_teacher(epoch)) and check_mean_teacher(epoch - 1) and not switched:
            model.eval()
            ema_model.eval()
            model.load_state_dict(ema_model.state_dict())
            switched = True
            print("SWITCHED!")
            validate(dataloader_test, model, ema_model, criterion, consistency_criterion, epoch)
            validate(dataloader_test, ema_model, model, criterion, consistency_criterion, epoch)
            model.train()
            ema_model.train()
        trainPacc, trainNacc, trainPNacc = train(dataloader_train_clean, dataloader_train_noisy, model, ema_model, criterion, consistency_criterion, optimizer, scheduler, epoch, self_paced_pick = len(dataset_train_clean))

        valPacc, valNacc, valPNacc = validate(dataloader_test, model, ema_model, criterion, consistency_criterion, epoch)
        val.append(valPNacc)
        #validate_2(dataloader_test, model, ema_model, criterion, consistency_criterion, epoch)
        stats_._update(trainPacc, trainNacc, trainPNacc, valPacc, valNacc, valPNacc)

        is_best = valPNacc > best_acc
        best_acc = max(valPNacc, best_acc)
        filename = []
        filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
        filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))

        if (check_self_paced(epoch)) and (epoch - args.self_paced_start) % args.self_paced_frequency == 0:

            dataloader_train_clean, dataloader_train_noisy = update_dataset(model, ema_model, dataset_train_clean, dataset_train_noisy, epoch)

        plot_curve(stats_, args.modeldir, 'model', True)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename)
        dataset_train_noisy.shuffle()

        #dataloader_train_clean = DataLoader(dataset_train_clean, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
        dataloader_train_noisy = DataLoader(dataset_train_noisy, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    print(best_acc)
    print(val)

def train(clean_loader, noisy_loader, model, ema_model, criterion, consistency_criterion, optimizer, scheduler,  epoch, warmup = False, self_paced_pick = 0):

    global step, switched, single_epoch_steps
    pacc = AverageMeter()
    nacc = AverageMeter()
    pnacc = AverageMeter()
    model.train()
    ema_model.train()
    consistency_weight = get_current_consistency_weight(epoch - 30)
    resultt = np.zeros(61000)
    if not warmup: scheduler.step()
    print("Learning rate is {}".format(optimizer.param_groups[0]['lr']))
    if clean_loader:
        for i, (X, _, Y, T, ids, _) in enumerate(clean_loader):
            # measure data loading time
            if args.gpu == None:
                X = X.cuda()
                Y = Y.cuda().float()
                T = T.cuda().long()
            else:
                X = X.cuda()
                Y = Y.cuda().float()
                T = T.cuda().long()
            if args.dataset == 'mnist':
                X = X.view(X.shape[0], 1, -1)
            

            # compute output
            output = model(X)
            with torch.no_grad():
                ema_output = ema_model(X)

            consistency_loss = consistency_weight * \
            consistency_criterion(output, ema_output) / X.shape[0]
            #if epoch >= args.self_paced_start: criterion.update_p(0.5)
            _, loss = criterion(output, Y) # 计算loss，使用PU标签
            
            #print(output)
            # measure accuracy and record loss
            
            if check_mean_teacher(epoch):
                predictions = torch.sign(ema_output).long() # 使用teacher的结果作为预测
            else:
                predictions = torch.sign(output).long() # 否则使用自己的结果

            smx = torch.sigmoid(output) # 计算sigmoid概率
            #print(smx)
            smx = torch.cat([1 - smx, smx], dim=1) # 组合成预测变量
            smxY = ((Y + 1) / 2).long() # 分类结果，0-1分类
            if args.soft_label:
                aux = - torch.sum(smx * torch.log(smx + 1e-10)) / smx.shape[0]
            else:
                smxY = smxY.float()
                smxY = smxY.view(-1, 1)
                smxY = torch.cat([1 - smxY, smxY], dim = 1)
                aux = - torch.sum(smxY * torch.log(smx + 1e-10)) / smxY.shape[0] # 计算Xent loss
                
            loss = aux
            if check_mean_teacher(epoch):
                loss += consistency_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if check_mean_teacher(epoch) and ((i + 1) % int(single_epoch_steps / 2 - 1)) == 0:
                update_ema_variables(model, ema_model, args.ema_decay, step) # 更新ema参数
                step += 1

            pacc_, nacc_, pnacc_, psize = accuracy(predictions, T) # 使用T来计算预测准确率
            pacc.update(pacc_, psize)
            nacc.update(nacc_, X.size(0) - psize)
            pnacc.update(pnacc_, X.size(0))

        print('Epoch Clean : [{0}]\t'
                'PACC {pacc.val:.3f} ({pacc.avg:.3f})\t'
                'NACC {nacc.val:.3f} ({nacc.avg:.3f})\t'
                'PNACC {pnacc.val:.3f} ({pnacc.avg:.3f})\t'.format(
                epoch, len(clean_loader), pacc=pacc, nacc=nacc, pnacc=pnacc))
    
    #if epoch > args.self_paced_start: criterion.update_p(0.05)
    #criterion.update_p(0.49)
    if args.update_pi:
        if (args.dataset == 'cifar'):
            #self_paced_pick = 0
            criterion.update_p((20000 - self_paced_pick / 2) / (50000 - self_paced_pick))
            print("Setting Pi_P to {}".format((20000 - self_paced_pick / 2) / (50000 - self_paced_pick)))
        elif args.dataset == 'mnist':
            criterion.update_p((30000 - self_paced_pick / 2) / (60000 - self_paced_pick))
            print("Setting Pi_P to {}".format((30000 - self_paced_pick / 2) / (60000 - self_paced_pick)))

    if epoch <= args.noisy_stop:
        for i, (X, Y, _, T, ids, p) in enumerate(noisy_loader):
            if args.gpu == None:
                X = X.cuda()
                Y = Y.cuda().float()
                T = T.cuda().long()
                p = p.cuda().float()
            else:
                X = X.cuda()
                Y = Y.cuda().float()
                T = T.cuda().long()
                p = p.cuda().float()
            if args.dataset == 'mnist':
                X = X.view(X.shape[0], 1, -1)

            output = model(X)

            with torch.no_grad():
                ema_output = ema_model(X)

            consistency_loss = consistency_weight * \
            consistency_criterion(output, ema_output) / X.shape[0]
            
            #if epoch >= args.self_paced_start: criterion.update_p(0.5)
            _, loss = criterion(output, Y)
            if check_mean_teacher(epoch) and not warmup:
                loss += consistency_loss
                predictions = torch.sign(ema_output).long()
            else:
                predictions = torch.sign(output).long()

            #if epoch >= args.self_paced_start

            if check_noisy(epoch):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if check_mean_teacher(epoch) and ((i + 1) % int(single_epoch_steps / 2 - 1)) == 0 and not warmup:
                update_ema_variables(model, ema_model, args.ema_decay, step)
                step += 1

            pacc_, nacc_, pnacc_, psize = accuracy(predictions, T)
            #print(torch.sum(Y==1))
            pacc.update(pacc_, psize)
            nacc.update(nacc_, X.size(0) - psize)
            pnacc.update(pnacc_, X.size(0))
        
        #print(results)
        #print(resultt)
        print('Noisy Epoch: [{0}]\t'
                'PACC {pacc.val:.3f} ({pacc.avg:.3f})\t'
                'NACC {nacc.val:.3f} ({nacc.avg:.3f})\t'
                'PNACC {pnacc.val:.3f} ({pnacc.avg:.3f})\t'.format(
                epoch, i, len(noisy_loader), pacc=pacc, nacc = nacc, pnacc=pnacc))

    
    return pacc.avg, nacc.avg, pnacc.avg

def validate(val_loader, model, ema_model, criterion, consistency_criterion, epoch):
    pacc = AverageMeter()
    nacc = AverageMeter()
    pnacc = AverageMeter()
    model.eval()
    ema_model.eval()
    consistency_weight = get_current_consistency_weight(epoch - 30)

    with torch.no_grad():
        for i, (X, Y, _, T, ids, p) in enumerate(val_loader):
            # measure data loading time

            if args.gpu == None:
                X = X.cuda()
                Y = Y.cuda().float()
                T = T.cuda().long()
            else:
                X = X.cuda()
                Y = Y.cuda().float()
                T = T.cuda().long()
            if args.dataset == 'mnist':
                X = X.view(X.shape[0], 1, -1)

            # compute output
            output = model(X)
            ema_output = ema_model(X)
            
            if check_mean_teacher(epoch):
                predictions = torch.sign(ema_output).long()      
            else:
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

def validate_2(val_loader, model, ema_model, criterion, consistency_criterion, epoch):
    pacc = AverageMeter()
    nacc = AverageMeter()
    pnacc = AverageMeter()
    model.eval()
    ema_model.eval()
    consistency_weight = get_current_consistency_weight(epoch - 30)
    
    with torch.no_grad():
        for i, (X, Y, _, T, ids) in enumerate(val_loader):
            # measure data loading time

            X = X.cuda()
            if args.dataset == 'mnist':
                X = X.view(X.shape[0], 1, -1)
            Y = Y.cuda().float()
            T = T.cuda().long()

            # compute output
            output = model(X)
            ema_output = ema_model(X)
            
            
            predictions = torch.sign(output).long()
            
            pacc_, nacc_, pnacc_, psize = accuracy(predictions, T)
            pacc.update(pacc_, psize)
            nacc.update(nacc_, X.size(0) - psize)
            pnacc.update(pnacc_, X.size(0))
            

    print('Test Student [{0}]: \t'
                'PACC {pacc.val:.3f} ({pacc.avg:.3f})\t'
                'NACC {nacc.val:.3f} ({nacc.avg:.3f})\t'
                'PNACC {pnacc.val:.3f} ({pnacc.avg:.3f})\t'.format(
                epoch, pacc=pacc, nacc=nacc, pnacc=pnacc))
    print("=====================================")

def create_model(ema=False):
    model = Model(28*28)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def create_cifar_model(ema=False):
    model = CNN()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def update_ema_variables(model, ema_model, alpha, global_step):

    alpha = min(1 - 1 / (global_step + 1), alpha)
    print(alpha)
    #print(alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        #print(torch.max(abs(ema_param.data - param.data)))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_criterion():
    weights = [float(args.weight), 1.0]
    class_weights = torch.FloatTensor(weights)

    class_weights = class_weights.cuda()
    if args.loss == 'Xent':
        criterion = PULoss(Probability_P=0.49, loss_fn="Xent")
    elif args.loss == 'nnPU':
        criterion = PULoss(Probability_P=0.49)
    elif args.loss == 'Focal':
        class_weights = torch.FloatTensor(weights).cuda()
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
        return torch.tensor(0.).cuda(), ncorrect / (batch_size - ptotal) * 100, correct / batch_size * 100, ptotal
    elif ptotal == batch_size:
        return pcorrect / ptotal * 100, torch.tensor(0.).cuda(), correct / batch_size * 100, ptotal
    else:
        return pcorrect / ptotal * 100, ncorrect / (batch_size - ptotal) * 100, correct / batch_size * 100, ptotal

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])

def update_dataset(student, teacher, dataset_train_clean, dataset_train_noisy, epoch, ratio=0.5, lt = 0, ht = 1):
    global results
    dataset_train_noisy.reset_ids()
    dataset_train_noisy.set_type("clean")
    dataloader_train = DataLoader(dataset_train_noisy, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)
    if args.dataset == 'mnist':
        results = np.zeros(61000) # rid.imageid: p_pos # 存储概率结果
    elif args.dataset == 'cifar':
        results = np.zeros(51000) 
    student.eval()
    teacher.eval()
    # validation #######################
    with torch.no_grad():
        for i_batch, (X, _, _, T, ids, _) in enumerate(tqdm(dataloader_train)):
            #images, lefts, rights, ages, genders, edus, apoes, labels, pu_labels, ids = Variable(sample_batched['mri']).cuda(), Variable(sample_batched['left']).cuda(), Variable(sample_batched['right']).cuda(), Variable(sample_batched['age']).cuda(), Variable(sample_batched['gender']).cuda(), Variable(sample_batched['edu']).cuda(), Variable(sample_batched['apoe']).cuda(), Variable(sample_batched['label']).view(-1).type(torch.LongTensor).cuda(), Variable(sample_batched['pu_label']).view(-1).type(torch.LongTensor).cuda(), sample_batched['id']
            X = X.cuda()
            if args.dataset == 'mnist':
                X = X.reshape(X.shape[0], 1, -1)
            # ===================forward====================
            outputs_s = student(X)
            outputs_t = teacher(X)
            prob_n_s = torch.sigmoid(outputs_s).view(-1).cpu().numpy()       
            prob_n_t = torch.sigmoid(outputs_t).view(-1).cpu().numpy()
            #print(np.sum(prob_n_t < 0.5))
            if check_mean_teacher(epoch):
                results[ids.view(-1).numpy()] = prob_n_t
            else:
                results[ids.view(-1).numpy()] = prob_n_s
    # adni_dataset_train.update_labels(results, ratio)
    # dataset_origin = dataset_train
    ids_noisy = dataset_train_clean.update_ids(results, epoch, ratio = ratio, ht = ht, lt = lt) # 返回的是noisy ids
    dataset_train_noisy.set_ids(ids_noisy) # 将noisy ids更新进去
    dataset_train_noisy.set_type("noisy")
    dataset_train_noisy.update_prob(results)
    dataset_train_clean.update_prob(results)
    assert np.all(dataset_train_noisy.ids == ids_noisy) # 确定更新了
    #dataloader_origin = DataLoader(dataset_origin, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    dataloader_train_clean = DataLoader(dataset_train_clean, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    dataloader_train_noisy = DataLoader(dataset_train_noisy, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    return dataloader_train_clean, dataloader_train_noisy

def check_mean_teacher(epoch):
    if not args.mean_teacher:
        return False
    elif epoch < args.ema_start:
        return False 
    else:
        return True

def check_self_paced(epoch):
    if args.self_paced_stop < 0: self_paced_stop = args.epochs
    else: self_paced_stop = args.self_paced_stop
    if not args.self_paced:
        return False
    elif args.self_paced and epoch >= self_paced_stop:
        return False
    elif args.self_paced and epoch < args.self_paced_start:
        return False
    else: return True

def check_noisy(epoch):
    if epoch >= args.self_paced_start and args.turnoff_noisy:
        return False
    else:
        return True


if __name__ == '__main__':
    main()