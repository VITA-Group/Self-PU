import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from sklearn import manifold
from sklearn.metrics import pairwise
from sklearn.utils import extmath


non_image_vars = ['Age', 'PTGENDER', 'PTEDUCAT', 'APOE Status', 'MMSCORE', 'CDR', 'AVLT-LTM', 'AVLT-Total', 'ADAS']
one_hot_vars = {"APOE Status": {'NC': 0, 'HT': 1, 'HM': 2, 0.0: 3}}
dx2label = {"AD": 0, "MCI": 1, "NL": 2}


def one_hot_torch(index, classes):
    '''
    index: labels, batch_size * 1, index starts from 0
    classes: int, # of classes
    '''
    y = index.type(torch.LongTensor)
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(y.size()[0], classes)
    y_onehot.zero_()
    '''
        TypeError: scatter_ received an invalid combination of arguments - got (int, Variable, int), but expected one of:
     * (int dim, torch.LongTensor index, float value)
          didn't match because some of the arguments have invalid types: (int, Variable, int)
     * (int dim, torch.LongTensor index, torch.FloatTensor src)
          didn't match because some of the arguments have invalid types: (int, Variable, int)
      '''
    y_onehot.scatter_(1, y.data, 1)
    #return Variable(y_onehot).cuda()
    return Variable(y_onehot)


def focal_loss(input, y, weight=None, alpha=0.25, gamma=2, eps=1e-7, reduction='elementwise_mean', one_hot=True, reverse_weighting=False):
    # print("focal loss:", input, target)
    y = y.view(-1, 1)

    ###############################
    if one_hot:
        y_hot = one_hot_torch(y, input.size(-1))
    else:
        #y_hot = Variable(torch.ones(y.size(0), 2)).cuda() * y # y is float tensor
        y_hot = Variable(torch.ones(y.size(0), 2).cuda()) * y
        y_hot[:, 0] = 1 - y_hot[:, 1]
    ###############################

    if weight is None:
        logit = F.softmax(input, dim=-1)
    else:
        logit = F.softmax(input, dim=-1) * weight
    logit = logit.clamp(eps, 1. - eps)

    loss = -1 * y_hot * torch.log(logit) # cross entropy
    if reverse_weighting:
        for i in range(loss.size()[0]):
            index = torch.argmax(y_hot)
            loss[i, index] = loss[i, index] * (1 - logit[i, 1 - index]) ** gamma
        loss *= alpha
    else:
        loss = alpha * loss * (1 - logit) ** gamma # focal loss

    if reduction == 'elementwise_mean':
        return None,loss.sum() / input.size()[0]
    elif reduction == 'sum':
        return None,loss.sum()
    elif reduction == 'elementwise_sum':
        return None,loss.sum(dim=1)
    else:
        return None,loss


class FocalLoss(nn.Module):

    def __init__(self, weight=None, alpha=0.25, gamma=2, eps=1e-7, one_hot=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.weight = weight
        self.one_hot = one_hot

    def forward(self, input, y):
        return focal_loss(input, y, weight=self.weight, alpha=self.alpha, gamma=self.gamma, eps=self.eps, one_hot=self.one_hot)

def crossentropy_loss(input):

    loss = -torch.log(torch.sigmoid(input))
    return loss

def sigmoid_loss(input, reduction='elementwise_mean'):
    # y must be -1/+1
    # NOTE: torch.nn.functional.sigmoid is 1 / (1 + exp(-x)). BUT sigmoid loss should be 1 / (1 + exp(x))
    loss = torch.sigmoid(-input)
    
    return loss

class SigmoidLoss(nn.Module):

    def __init__(self, reduction='elementwise_mean'):
        super(SigmoidLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, y):
        return sigmoid_loss(input, y, self.reduction)


def edge_weight(In_data): 
    Rho = 1e-2
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    X = extmath.row_norms(In_data, squared=True) # Row-wise (squared) Euclidean norm of X.
    X = X[:,np.newaxis]
    kernel = np.dot(In_data, In_data.T)
    XX = np.ones((len(X), 1))
    X = np.dot(X, XX.T)
    kernel *= -2
    kernel = X + kernel + X.T
    kernel = np.exp(-Rho * kernel)
    return kernel


def laplacian(In_data, normal=False):
    In_data = In_data.reshape(len(In_data), -1)
    # In_data = np.float128(In_data)/255.
    adj_mat = edge_weight(In_data)
    D = np.zeros((len(In_data), len(In_data)))
    for n in range(len(D)):
        D[n,n] = np.sum(adj_mat[n,:])
    if normal == True:
        sqrt_deg_matrix = np.mat(np.diag(np.diag(D)**(-0.5)))
        lap_matrix = sqrt_deg_matrix * np.mat(D - adj_mat) * sqrt_deg_matrix
    else:
        lap_matrix = D - adj_mat
    return (np.float32(lap_matrix))

def pu_risk_estimators_sigmoid(y_pred, y_true, prior):
    # y_true is -1/1
    #one_u = torch.ones(y_true.size()).cuda()
    one_u = torch.ones(y_true.size())
    positive = (y_true == 1).float()
    unlabeled = (y_true == -1).float()
    P_size = max(1., torch.sum(positive))
    u_size = max(1. ,torch.sum(unlabeled))
    y_positive = sigmoid_loss(y_pred).view(-1)
    y_unlabeled = sigmoid_loss(-y_pred).view(-1)
    positive_risk = (prior * y_positive * positive / P_size).sum()
    negative_risk = ((unlabeled / u_size - prior * positive / P_size) * y_unlabeled).sum()
    return positive_risk, negative_risk
def pu_risk_estimators_sigmoid_eps(y_pred, y_true, prior, eps):
    # y_true is -1/1
    #one_u = torch.ones(y_true.size()).cuda()
    one_u = torch.ones(y_true.size())
    positive = (y_true == 1).float()
    unlabeled = (y_true == -1).float()
    P_size = max(1., torch.sum(positive))
    u_size = max(1. ,torch.sum(unlabeled))
    y_positive = sigmoid_loss(y_pred).view(-1) * eps
    y_unlabeled = sigmoid_loss(-y_pred).view(-1) * eps
    positive_risk = ((prior * y_positive * positive / P_size)).sum()
    negative_risk = (((unlabeled / u_size - prior * positive / P_size) * y_unlabeled)).sum()
    return positive_risk, negative_risk

def nu_risk_estimators_sigmoid(y_pred, y_true, prior):
    # y_true is -1/1
    #one_u = torch.ones(y_true.size()).cuda()
    one_u = torch.ones(y_true.size())
    positive = (y_true == 1).float()
    unlabeled = (y_true == -1).float()
    P_size = max(1., torch.sum(positive))
    u_size = max(1. ,torch.sum(unlabeled))
    y_positive = sigmoid_loss(y_pred).view(-1)
    y_unlabeled = sigmoid_loss(-y_pred).view(-1)
    positive_risk = (prior * y_positive * positive / P_size).sum()
    negative_risk = ((unlabeled / u_size - prior * positive / P_size) * y_unlabeled).sum()
    return positive_risk, negative_risk

def pu_risk_estimators_crossentropy(y_pred, y_true, prior):

    one_u = torch.ones(y_true.size())
    positive = (y_true == 1).float()
    unlabeled = (y_true == -1).float()
    P_size = max(1., torch.sum(positive))
    u_size = max(1. ,torch.sum(unlabeled))
    y_positive = crossentropy_loss(y_pred).view(-1)
    y_unlabeled = crossentropy_loss(-y_pred).view(-1)
    positive_risk = (y_positive * positive / P_size).sum()
    negative_risk = ((unlabeled / u_size) * y_unlabeled).sum()
    #print(P_p, P_n, P_u)
    return positive_risk, negative_risk
def pu_risk_estimators_focal(y_pred, y_true):
    # y_pred is [score1, score2] before softmax logit, y_true is 0/1
    #one_u = torch.ones(y_true.size()).cuda()
    #zeros = torch.zeros(y_true.size()).cuda()
    one_u = torch.ones(y_true.size())
    zeros = torch.zeros(y_true.size())
    u_mask = torch.abs(y_true - one_u)
    
    #P_size = torch.max(torch.sum(y_true), torch.Tensor([1]).cuda())
    P_size = torch.max(torch.sum(y_true), torch.Tensor([1]))
    #u_size = torch.max(torch.sum(u_mask), torch.Tensor([1]).cuda())
    u_size = torch.max(torch.sum(u_mask), torch.Tensor([1]))
    P_p = (focal_loss(y_pred, one_u, gamma=3, reduction='elementwise_sum')).dot(y_true) / P_size # should go down
    P_n = (focal_loss(y_pred, zeros, gamma=3, reduction='elementwise_sum')).dot(y_true) / P_size # should go up
    P_u = (focal_loss(y_pred, zeros, gamma=3, reduction='elementwise_sum')).dot(u_mask) / u_size # should go down
    return P_p, P_n, P_u


def pu_loss(y_pred, y_true, loss_fn, Probility_P=0.25, BETA=0, gamma=1.0, Yi=1e-8, L=None, nnPU = True, eps = None):
    P_p, P_n, P_u = 0, 0, 0
    if loss_fn == "sigmoid":
        R_p, R_n = pu_risk_estimators_sigmoid(y_pred, y_true, Probility_P)
    elif loss_fn == "focal":
        P_p, P_n, P_u = pu_risk_estimators_focal(y_pred, y_true)
    elif loss_fn == 'Xent':
        R_p, R_n = pu_risk_estimators_crossentropy(y_pred, y_true, Probility_P)
    elif loss_fn == 'sigmoid_eps':
        R_p, R_n = pu_risk_estimators_sigmoid_eps(y_pred, y_true, Probility_P, eps)
    else: pass

    M_reg = torch.zeros(1)
    if L is not None:
        FL = torch.mm((2 * y_pred - 1).transpose(0, 1), L)
        R_manifold = torch.mm(FL, (2 * y_pred - 1))
        M_reg = Yi * R_manifold
    if (not nnPU) or (loss_fn == 'Xent'):
        return None, R_p + R_n 

    if -BETA > R_n:
        #print("NEGATIVE")
        #print(R_n)
        return R_p - BETA, -gamma*R_n#, Probility_P * P_p, P_u, Probility_P * P_
        # return -gamma * PU_2, torch.sum(M_reg), Probility_P * P_p, P_u, Probility_P * P_n
        # return Probility_P * P_p
    else:
        #print("POSITIVE")
        #print(R_p, R_n, R_p + R_n)
        return R_p + R_n, R_p + R_n#, Probility_P * P_p, P_u, Probility_P * P_n
        # return PU_1, torch.sum(M_reg), Probility_P * P_p, P_u, Probility_P * P_n


class PULoss(nn.Module):
    '''
    only works for binary classification
    '''

    def __init__(self, loss_fn='sigmoid', Probability_P=0.25, BETA=0, gamma=1.0, Yi=1e-8, nnPU=True):
        super(PULoss, self).__init__()
        self.loss_fn = loss_fn
        self.Probability_P = Probability_P
        self.BETA = BETA
        self.gamma = gamma
        self.Yi = Yi
        self.nnPU = nnPU

    def update_p(self, p):
        self.Probability_P = p
    def forward(self, y_pred, y_true, L=None, eps = None):
        return pu_loss(y_pred, y_true, self.loss_fn, self.Probability_P, self.BETA, self.gamma, self.Yi, L, nnPU = self.nnPU, eps = eps)



def L1_reg(model):
    # https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = W.norm(1)
        else:
            l1_reg = l1_reg + W.norm(1)
    return l1_reg


def suvr2class(suvrs):
    labels = torch.round((suvrs - 0.8) * 10).type(torch.LongTensor)
    return labels




def show_slices(slices, lower = None, upper = None):
    fig, axes = plt.subplots(1, len(slices), figsize=(30,30))
    for i, slice in enumerate(slices):
        if lower != None and upper != None: axes[i].imshow(slice.T, cmap="gray", origin="lower", vmin=lower, vmax=upper)
        elif lower != None: axes[i].imshow(slice.T, cmap="gray", origin="lower", vmin=lower)
        elif upper != None: axes[i].imshow(slice.T, cmap="gray", origin="lower", vmax=upper)
        else: axes[i].imshow(slice.T, cmap="gray", origin="lower")



def confusion_matrix(predictions, truths, classes):
    '''
    predictions, truths: list of integers
    classes: int, # of classes
    return confusion_matrix: x-axis target, y-axis predictions
    '''
    m = np.zeros((classes, classes))
    accuracy = np.zeros(classes)
    for i in range(len(predictions)):
        m[int(predictions[i]), int(truths[i])] += 1
    diagonal = 0
    for i in range(classes):
        accuracy[i] = m[i, i] / np.sum(m[:, i], axis=0)
        diagonal += m[i, i]
    return m, accuracy, float(diagonal) / len(predictions)

class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

