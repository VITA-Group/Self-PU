import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from mean_teacher import losses, ramps
from utils.util import FocalLoss, PULoss
from models import MultiLayerPerceptron as Model
from models import CNN
from datasets import MNIST_Dataset_FixSample, get_mnist, binarize_mnist_class
from cifar_datasets import CIFAR_Dataset, get_cifar, binarize_cifar_class
from functions import *
from torchvision import transforms

import os
import time
import random
import argparse
import numpy as np
import shutil

from tqdm import tqdm

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, help='workers')


parser.add_argument('--self-paced', type=boolean_string, default=True)
parser.add_argument('--self-paced-start', type=int, default=10)
parser.add_argument('--self-paced-stop', type=int, default=50)
parser.add_argument('--self-paced-frequency', type=int, default=10)
parser.add_argument('--self-paced-type', type=str, default = "A")

parser.add_argument('--increasing', type=boolean_string, default=True)
parser.add_argument('--replacement', type=boolean_string, default=True)

parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--datapath', type=str, default="")
parser.add_argument('--model', type=str, default=None)


step = 0
results = np.zeros(61000)
switched = False
results1 = None
results2 = None
args = None

def main():

    global args, switched
    args = parser.parse_args()

    print(args)
    criterion = get_criterion()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.dataset == "mnist":
        (trainX, trainY), (testX, testY) = get_mnist()
        _trainY, _testY = binarize_mnist_class(trainY, testY)

        dataset_test = MNIST_Dataset_FixSample(args.log_path, args.task_name, 1000, 60000, 
            trainX, _trainY, testX, _testY, split='test',
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, type="clean",
        seed = args.seed)

    elif args.dataset == 'cifar':
        (trainX, trainY), (testX, testY) = get_cifar()
        _trainY, _testY = binarize_cifar_class(trainY, testY)

        dataset_test = CIFAR_Dataset(args.log_path, args.task_name, 1000, 50000, 
            trainX, _trainY, testX, _testY, split='test',
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, transform = data_transforms['val'], type="clean",
        seed = args.seed)

        criterion.update_p(0.4)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    consistency_criterion = losses.softmax_mse_loss
    if args.dataset == 'mnist':
        model = create_model()
    elif args.dataset == 'cifar':
        model = create_cifar_model()
    if args.gpu is not None:
        model = model.cuda()
    else:
        model = model.cuda()

    stats_ = stats(args.modeldir, 0)
    print("Evaluation mode!")
    
    if args.model is None:
        raise RuntimeError("Please specify a model file.")
    else:
        state_dict = torch.load(args.model)['state_dict']
        model.load_state_dict(state_dict)

    valPacc, valNacc, valPNacc = validate(dataloader_test, model)

def validate(val_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pacc = AverageMeter()
    nacc = AverageMeter()
    pnacc
    model.eval()
    end = time.time()
    
    with torch.no_grad():
        for i, (X, Y, _, T, ids, _) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            X = X.cuda(args.gpu)
            if args.dataset == 'mnist':
                X = X.view(X.shape[0], 1, -1)
            Y = Y.cuda(args.gpu).float()
            T = T.cuda(args.gpu).long()

            # compute output
            output = model(X)
            prediction = torch.sign(output).long()
            
            pacc_, nacc_, pnacc_, psize = accuracy(predictions, T)
            pacc.update(pacc_, X.size(0))
            nacc.update(nacc_, X.size(0))
            pnacc.update(pnacc_, X.size(0))

    print('Test [{0}]: \t'
                'PNACC {pnacc1.val:.3f} ({pnacc1.avg:.3f})\t'
                'PNACC2 {pnacc2.val:.3f} ({pnacc2.avg:.3f})\t'
                'PNACC3 {pnacc3.val:.3f} ({pnacc3.avg:.3f})\t'
                'PNACC4 {pnacc4.val:.3f} ({pnacc4.avg:.3f})\t'.format(
                epoch, pnacc1=pnacc1, pnacc2=pnacc2, pnacc3=pnacc3, pnacc4 = pnacc4))
    print("=====================================")
    return pacc.avg, nacc.avg, pnacc1.avg, pnacc2.avg, pnacc3.avg , pnacc4.avg

def create_model():
    model = Model(28*28)
    return model

def create_cifar_model():
    model = CNN()
    return model

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

if __name__ == '__main__':
    main()