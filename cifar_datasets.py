import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging
import os
import copy
import os.path as osp
import pickle
from collections import Counter


def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result

def _load_datafile(filename):
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        #print(data_dict.keys())
        assert data_dict[b'data'].dtype == np.uint8
        image_data = data_dict[b'data']
        image_data = image_data.reshape((image_data.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
        return image_data, np.array(data_dict[b'labels'])

def get_cifar():
    train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
    eval_filename = 'test_batch'

    x_tr = np.zeros((50000, 32, 32, 3), dtype='uint8')
    y_tr = np.zeros(50000, dtype='int32')

    for ii, fname in enumerate(train_filenames):
        cur_images, cur_labels = _load_datafile(osp.join('cifar', fname))
        x_tr[ii * 10000 : (ii+1) * 10000, ...] = cur_images
        y_tr[ii * 10000 : (ii+1) * 10000, ...] = cur_labels

    x_te, y_te = _load_datafile(osp.join('cifar', eval_filename))
    return (x_tr, y_tr), (x_te, y_te)


def binarize_cifar_class(_trainY, _testY):
    trainY = - np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY == 0] = 1
    trainY[_trainY == 1] = 1
    trainY[_trainY == 8] = 1
    trainY[_trainY == 9] = 1
    testY = - np.ones(len(_testY), dtype=np.int32)
    testY[_testY == 0] = 1
    testY[_testY == 1] = 1
    testY[_testY == 8] = 1
    testY[_testY == 9] = 1
    return trainY, testY

def make_cifar_dataset(dataset, n_labeled, n_unlabeled, mode="train", pn=False, seed = None):
    def make_PU_dataset_from_binary_dataset(x, y, labeled=n_labeled, unlabeled=n_unlabeled):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.uint8), np.asarray(y, dtype=np.int32)
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(X))
        X, Y = X[perm], Y[perm]#, realy[perm]
        assert(len(X) == len(Y))
        n_p = (Y == positive).sum()
        n_lp = labeled
        n_n = (Y == negative).sum()
        n_u = unlabeled
        if labeled + unlabeled == len(X):
            n_up = n_p - n_lp
        elif unlabeled == len(X):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        prior = float(n_up) / float(n_u)
        Xlp = X[Y == positive][:n_lp]
        #rlp = realy[Y == positive][:n_lp]
        Xup = np.concatenate((X[Y == positive][n_lp:], Xlp), axis=0)[:n_up]
        #rup = np.concatenate((realy[Y == positive][n_lp:], rlp), axis=0)[:n_up]
        Xun = X[Y == negative]
        #run = realy[Y == negative]
        X = np.asarray(np.concatenate((Xlp, Xup, Xun), axis=0), dtype=np.uint8)
        Y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))), dtype=np.int32)
        #print(all_np(rlp))
        T = np.asarray(np.concatenate((np.ones(n_lp + n_up), -np.ones(n_u-n_up))), dtype=np.int32)
        ### Generate ID
        ids = np.array([i for i in range(len(X))])
        return X, Y, T, ids, prior

    def make_PN_dataset_from_binary_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.uint8), np.asarray(y, dtype=np.int32)
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(X))
        X, Y = X[perm], Y[perm]
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        Xp = X[Y == positive][:n_p]
        Xn = X[Y == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.uint8)
        Y = np.asarray(np.concatenate((np.ones(n_p), -np.ones(n_n))), dtype=np.int32)
        ids = np.array([i for i in range(len(X))])
        return X, Y, Y, ids
    def make_only_PN_train(x,y, n_labeled=n_labeled, prior=0.5):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.uint8), np.asarray(y, dtype=np.int32)
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(X))
        X, Y = X[perm], Y[perm]
        assert(len(X) == len(Y))
        n_n = int(n_labeled * pow(prior / (2 * (1-prior)), 2))
        Xp = X[Y == positive][:n_labeled]
        Xn = X[Y == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.uint8)
        Y = np.asarray(np.concatenate((np.ones(n_labeled), -np.ones(n_n))), dtype=np.int32)
        ids = np.array([i for i in range(len(X))])
        
        return X, Y, Y, ids

    (_trainX, _trainY), (_testX, _testY) = dataset
    prior = None
    if (mode == 'train'):
        if not pn:
            X, Y, T, ids, prior = make_PU_dataset_from_binary_dataset(_trainX, _trainY)
        else:
            X, Y, T, ids = make_only_PN_train(_trainX, _trainY)
    else:
        X, Y, T, ids  = make_PN_dataset_from_binary_dataset(_testX, _testY)
    #print("training:{}".format(trainX.shape))
    #print("test:{}".format(testX.shape))
    return X, Y, T, ids, prior

class CIFAR_Dataset(Dataset):

    def __init__(self, n_labeled, n_unlabeled, trainX, trainY, testX, testY, type="noisy", split="train", mode="N", ids=None, pn=False, increasing=False, replacement=True, top = 0.5, transform = None, flex = 0, pickout = True, seed = None):

        self.X, self.Y, self.T, self.oids, self.prior = make_cifar_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
        assert np.all(self.oids == np.linspace(0, len(self.X) - 1, len(self.X)))
        self.clean_ids = []
        self.P = self.Y.copy()
        if (ids is None):
            self.ids = self.oids
        else:
            self.ids = np.array(ids)

        self.split = split
        self.mode = mode
        self.type = type
        self.pos_ids = self.oids[self.Y == 1]
        self.pid = self.pos_ids
        if len(self.ids) != 0:
            self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
        else:
            self.uid = []

        self.sample_ratio = len(self.uid) // len(self.pid)  + 1
        print("origin:", len(self.pos_ids), len(self.ids))
        self.increasing = increasing
        self.replacement = replacement
        self.top = top
        self.transform = transform
        self.flex = flex
        self.pickout = pickout

        self.pick_accuracy = []
    def copy(self, dataset):
        ''' Copy random sequence
        '''
        self.X, self.Y, self.T, self.oids = dataset.X.copy(), dataset.Y.copy(), dataset.T.copy(), dataset.oids.copy()
        self.P = self.Y.copy()
    def __len__(self):
        if self.type != 'noisy':
            return len(self.ids)
        else:
            return len(self.pid) * self.sample_ratio

    def set_type(self, type):
        self.type = type

    def shuffle(self):
        perm = np.random.permutation(len(self.uid))
        self.uid = self.uid[perm]

        perm = np.random.permutation(len(self.pid))
        self.pid = self.pid[perm]

    def __getitem__(self, idx): 
        # self.ids[idx]是真实的行索引
        # 始终使用真实的行索引去获得数据

        # 1901 保持比例
        if self.type == 'noisy':
            if (idx % self.sample_ratio == 0):
                index = self.pid[idx // self.sample_ratio]
                id = 0
            else:
                index = self.uid[idx - (idx // self.sample_ratio + 1)]
                #print(idx - idx // self.sample_ratio)
            return self.transform(self.X[index]), self.Y[index], self.P[index], self.T[index], index, 0
        else:
            return self.transform(self.X[self.ids[idx]]), self.Y[self.ids[idx]], self.P[self.ids[idx]], self.T[self.ids[idx]], self.ids[idx], 0

    def reset_ids(self):
        ''' Using all origin ids
        '''
        self.ids = self.oids.copy()

    def set_ids(self, ids):
        ''' Set specific ids
        '''
        self.ids = np.array(ids).copy()
        if len(ids) > 0:
            self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
            self.pid = np.intersect1d(self.ids[self.Y[self.ids] == 1], self.ids)
            self.sample_ratio = int(len(self.uid) / len(self.pid)) + 1

    def reset_labels(self):
        ''' Reset Y labels
        '''
        self.P = self.Y.copy()
    def update_prob(self, prob):
        pass

    def update_ids(self, results, epoch, ratio=None, ht = 0, lt = 0):
        if not self.replacement or self.increasing:
            percent = min(epoch / 100, 1) # 决定抽取数据的比例
        else:
            percent = 1

        if ratio == None:
            ratio = self.prior

        if self.mode == 'N':
            self.reset_labels()
            n_neg = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent) # 决定抽取的数量
            if self.replacement:
                # 如果替换的话，抽取n_neg个
                neg_ids = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)[:n_neg]
            else:
                # 否则抽取n_neg - #ids
                neg_ids = np.setdiff1d(np.argsort(results), self.ids, assume_unique=True)[:n_neg]
            # 变成向量
            neg_ids = np.array(neg_ids) 
            neg_label = self.T[neg_ids] # 获得neg_ids的真实标签
            correct = np.sum(neg_label < 1) # 抽取N的时候真实标签为-1
            
            print("Correct: {}/{}".format(correct, len(neg_ids))) # 打印
            if self.replacement:
                self.ids = np.concatenate([self.pos_ids[:len(self.pos_ids) // 2], neg_ids]) # 如果置换的话，在ids的基础上加上neg_ids
            else:
                if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids[:len(self.pos_ids) // 2]]) # 如果为空的话则首先加上pos_ids
                self.ids = np.concatenate([self.ids, neg_ids])

            self.ids = self.ids.astype(int) # 为了做差集

            out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
            assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
            return out

        elif self.mode == 'P':
            self.reset_labels()
            n_pos = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent) # 决定抽取的数量

            if self.replacement:
                # 如果替换的话，抽取n_neg个
                neg_ids = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)[-n_pos:]
            else:
                # 否则抽取n_neg - #ids
                neg_ids = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)[-(n_pos - len(self.ids)):]

            # 变成向量
            pos_ids = np.array(pos_ids) 
            pos_label = self.T[pos_ids] # 获得neg_ids的真实标签
            correct = np.sum(pos_label == 1) # 抽取N的时候真实标签为-1

            self.Y[pos_ids] = 1 # 将他们标注为1
            print("Correct: {}/{}".format(correct, len(pos_ids))) # 打印
            if self.replacement:
                self.ids = np.concatenate([self.pos_ids[:len(self.pos_ids) // 2], pos_ids]) # 如果置换的话，在ids的基础上加上neg_ids
            else:
                if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids[:len(self.pos_ids) // 2]]) # 如果为空的话则首先加上pos_ids
                self.ids = np.concatenate([self.ids, pos_ids])
            
            self.ids = self.ids.astype(int) # 为了做差集
            out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
            assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
            return out

        elif self.mode == 'A':
            self.reset_labels()
            n_all = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent * self.top) # 决定抽取的数量
            confident_num = int(n_all * (1 - self.flex))
            noisy_num = int(n_all * self.flex)
            if self.replacement:
                # 如果替换的话，抽取n_pos个
                #print(np.argsort(results))
                #print(np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True))
                al = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)
                neg_ids = al[:confident_num]
                pos_ids = al[-confident_num:]
            else:
                # 否则抽取n_pos - #ids
                al = np.setdiff1d(np.argsort(results), self.ids, assume_unique=True)
                neg_ids = al[:(confident_num - len(self.ids) // 2)]
                pos_ids = al[-(confident_num - len(self.ids) // 2):]

            # 变成向量
            pos_ids = np.array(pos_ids) 
            pos_label = self.T[pos_ids] # 获得neg_ids的真实标签
            pcorrect = np.sum(pos_label == 1) # 抽取N的时候真实标签为-1

            neg_ids = np.array(neg_ids) 
            neg_label = self.T[neg_ids] # 获得neg_ids的真实标签
            ncorrect = np.sum(neg_label < 1)
            self.pick_accuracy.append((pcorrect + ncorrect) * 1.0 / (len(pos_ids) * 2))
            self.P[pos_ids] = 1 # 将他们标注为1
            print("P Correct: {}/{}".format(pcorrect, len(pos_ids))) # 打印
            print("N Correct: {}/{}".format(ncorrect, len(neg_ids)))
            if self.replacement:
                #self.ids = np.concatenate([self.pos_ids, pos_ids, neg_ids]) # 如果置换的话，在ids的基础上加上neg_ids
                self.ids = np.concatenate([pos_ids, neg_ids]) 
            else:
                #if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids]) # 如果为空的话则首先加上pos_ids
                #self.ids = np.concatenate([self.ids, pos_ids, neg_ids])
                self.ids = np.concatenate([self.ids, pos_ids, neg_ids])
            
            self.ids = self.ids.astype(int) # 为了做差集
            if self.pickout:
                out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
            else:
                out = self.oids
            if noisy_num > 0:
                noisy_select = out[np.random.permutation(len(out))][:noisy_num]
                self.P[np.intersect1d(results >= 0.5, noisy_select)] = 1
                self.ids = np.concatenate([self.ids, noisy_select], 0)
                if self.pickout:
                    out = np.setdiff1d(self.oids, self.ids)
            if self.pickout:
                assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
            return out

        elif self.mode == 'E':
            self.reset_labels()
            n_all = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent * self.top) # 决定抽取的数量
            confident_num = int(n_all * (1 - self.flex))
            noisy_num = int(n_all * self.flex)

            al = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)
            ids = al[:confident_num]

            # 变成向量
            
            if self.replacement:
                #self.ids = np.concatenate([self.pos_ids, pos_ids, neg_ids]) # 如果置换的话，在ids的基础上加上neg_ids
                self.ids = ids
            else:
                #if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids]) # 如果为空的话则首先加上pos_ids
                #self.ids = np.concatenate([self.ids, pos_ids, neg_ids])
                self.ids = np.concatenate([self.ids, ids])
            
            self.ids = self.ids.astype(int) # 为了做差集
            if self.pickout:
                out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
            else:
                out = self.oids
            if noisy_num > 0:
                noisy_select = out[np.random.permutation(len(out))][:noisy_num]
                self.P[np.intersect1d(results >= 0.5, noisy_select)] = 1
                self.ids = np.concatenate([self.ids, noisy_select], 0)
                if self.pickout:
                    out = np.setdiff1d(self.oids, self.ids)
            if self.pickout:
                assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
            return out




