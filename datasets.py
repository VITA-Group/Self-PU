import os
import numpy as np
from sklearn.datasets import fetch_openml
from torch.utils.data import Dataset
import logging


def get_mnist():
    mnist_data = fetch_openml("mnist_784")
    x = mnist_data["data"]
    y = mnist_data["target"]
    # reshape to (#data, #channel, width, height)
    x = np.reshape(x, (x.shape[0], 1, 28, 28)) / 255.
    x_tr = np.asarray(x[:60000], dtype=np.float32)
    y_tr = np.asarray(y[:60000], dtype=np.int32)
    x_te = np.asarray(x[60000:], dtype=np.float32)
    y_te = np.asarray(y[60000:], dtype=np.int32)
    return (x_tr, y_tr), (x_te, y_te)


def binarize_mnist_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY % 2 == 1] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[_testY % 2 == 1] = -1
    return trainY, testY


class PNMNIST(Dataset):
    def __init__(self, split):
        (self.x_tr, self.y_tr), (self.x_te, self.y_te) = get_mnist()       
        self.y_tr, self.y_te = binarize_mnist_class(self.y_tr, self.y_te) 
        self.split = split
        print(self.x_te.shape)
        print(self.x_tr.shape)
        print(self.y_te.shape)
        print(self.y_tr.shape)

    def __len__(self):
        if (self.split == 'train'): 
            return self.x_tr.shape[0]
        else: 
            return self.x_te.shape[0]

    def __getitem__(self, index):
        if (self.split == 'train'):
            return self.x_tr[index], self.y_tr[index]

        else:
            return self.x_te[index], self.y_te[index]
        

def make_dataset(dataset, n_labeled, n_unlabeled, mode="train", pn=False, seed = None):
        
    def make_PU_dataset_from_binary_dataset(x, y, labeled=n_labeled, unlabeled=n_unlabeled):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(X))
        X, Y = X[perm], Y[perm]
        assert(len(X) == len(Y))

        n_p = (Y == positive).sum()
        n_lp = labeled
        n_u = unlabeled
        if labeled + unlabeled == len(X):
            n_up = n_p - n_lp
        elif unlabeled == len(X):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        prior = float(n_up) / float(n_u)
        Xlp = X[Y == positive][:n_lp]
        Xup = np.concatenate((X[Y == positive][n_lp:], Xlp), axis=0)[:n_up]
        Xun = X[Y == negative]
        X = np.asarray(np.concatenate((Xlp, Xup, Xun), axis=0), dtype=np.float32)
        Y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))), dtype=np.int32)
        T = np.asarray(np.concatenate((np.ones(n_lp + n_up), -np.ones(n_u-n_up))), dtype=np.int32)
        # Generate ID
        ids = np.array([i for i in range(len(X))])
        return X, Y, T, ids, prior

    def make_PN_dataset_from_binary_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(X))
        X, Y = X[perm], Y[perm]
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        Xp = X[Y == positive][:n_p]
        Xn = X[Y == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.float32)
        Y = np.asarray(np.concatenate((np.ones(n_p), -np.ones(n_n))), dtype=np.int32)
        ids = np.array([i for i in range(len(X))])
        return X, Y, Y, ids
    
    def make_only_PN_train(x, y, n_labeled=n_labeled, prior=0.5):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(X))
        X, Y = X[perm], Y[perm]
        assert(len(X) == len(Y))
        n_n = int(n_labeled * pow(prior / (2 * (1-prior)), 2))
        Xp = X[Y == positive][:n_labeled]
        Xn = X[Y == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.float32)
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
    return X, Y, T, ids, prior

class MNIST_Dataset(Dataset):

    def __init__(self, logpath, taskname, n_labeled, n_unlabeled, trainX, trainY, testX, testY, type="noisy", split="train", mode="N", ids=None, pn=False, increasing=False, replacement=True, top = 0.5, flex = 0, pickout=True, seed = None):

        self.X, self.Y, self.T, self.oids, self.prior = make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
        assert np.all(self.oids == np.linspace(0, len(self.X) - 1, len(self.X)))
        self.clean_ids = []
        # self.Y_origin = self.Y
        self.P = self.Y.copy()
        self.type = type
        if (ids is None):
            self.ids = self.oids
        else:
            self.ids = np.array(ids)

        self.split = split
        self.mode = mode
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        logname = logpath + taskname + "_dataset.log"
        if not os.path.exists(logpath):
            os.mkdir(logpath)
        fh = logging.FileHandler(logname, mode="w")
        self.logger.addHandler(fh)
        self.pos_ids = self.oids[self.Y == 1]

        self.pid = self.pos_ids
        if len(self.ids) != 0:
            self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
        else:
            self.uid = []
        print(len(self.uid))
        print(len(self.pid))
        self.sample_ratio = len(self.uid) // len(self.pid)  + 1
        print(self.sample_ratio)
        print("origin:", len(self.pos_ids), len(self.ids))
        self.increasing = increasing
        self.replacement = replacement
        self.top = top
        self.flex = flex
        self.pickout = pickout

        self.pick_accuracy = []
        self.result = -np.ones(len(self))

    def copy(self, dataset):
        ''' Copy random sequence
        '''
        self.X, self.Y, self.T, self.oids = dataset.X.copy(), dataset.Y.copy(), dataset.T.copy(), dataset.oids.copy()
        self.P = self.Y.copy()
        
    def __len__(self):
        if self.type == 'noisy':

            return len(self.pid) * self.sample_ratio
        else:
            return len(self.ids)

    def set_type(self, type):
        self.type = type

    def update_prob(self, result):
        rank = np.empty_like(result)
        rank[np.argsort(result)] = np.linspace(0, 1, len(result))
        #print(rank)
        if (len(self.pos_ids) > 0):
            rank[self.pos_ids] = -1
        self.result = rank
        
    def shuffle(self):
        perm = np.random.permutation(len(self.uid))
        self.uid = self.uid[perm]

        perm = np.random.permutation(len(self.pid))
        self.pid = self.pid[perm]

    def __getitem__(self, idx): 
        #print(idx)
        # self.ids[idx]是真实的行索引
        # 始终使用真实的行索引去获得数据

        # 1901 保持比例
        if self.type == 'noisy':
            if (idx % self.sample_ratio == 0):
                try:
                    index = self.pid[idx // self.sample_ratio]
                    id = self.ids[idx // self.sample_ratio]
                except IndexError:
                    print(idx)
                    print(self.sample_ratio)
                    print(len(self.pid))
            else:
                index = self.uid[idx - (idx // self.sample_ratio + 1)]
                id = self.ids[idx - (idx // self.sample_ratio + 1)]
            return self.X[index], self.Y[index], self.P[index], self.T[index], id, self.result[index]
        else:
            return self.X[self.ids[idx]], self.Y[self.ids[idx]], self.P[self.ids[idx]], self.T[self.ids[idx]], self.ids[idx], self.result[self.ids[idx]]


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
            if len(self.pid) == 0:
                self.sample_ratio = 10000000000
            else:
                self.sample_ratio = int(len(self.uid) / len(self.pid)) + 1
    def reset_labels(self):
        ''' Reset Y labels
        '''
        self.P = self.Y.copy()

    def update_ids(self, results, epoch, ratio=None, lt = 0, ht = 0):
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

            self.logger.info(correct) # 记录
            self.logger.info(neg_ids) # 记录
            print("Correct: {}/{}".format(correct, len(neg_ids))) # 打印
            if self.replacement:
                self.ids = np.concatenate([self.pos_ids[:len(self.pos_ids) // 2], neg_ids]) # 如果置换的话，在ids的基础上加上neg_ids
            else:
                if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids[:len(self.pos_ids) // 2]]) # 如果为空的话则首先加上pos_ids
                self.ids = np.concatenate([self.ids, neg_ids])

            self.ids = self.ids.astype(int) # 为了做差集
            out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
            #assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
            return out

        elif self.mode == 'P':
            self.reset_labels()
            n_pos = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent) # 决定抽取的数量

            if self.replacement:
                # 如果替换的话，抽取n_neg个
                pos_ids = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)[-n_pos:]
            else:
                # 否则抽取n_neg - #ids
                pos_ids = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)[-(n_pos - len(self.ids)):]

            # 变成向量
            pos_ids = np.array(pos_ids) 
            pos_label = self.T[pos_ids] # 获得neg_ids的真实标签
            correct = np.sum(pos_label == 1) # 抽取N的时候真实标签为-1

            self.Y[pos_ids] = 1 # 将他们标注为1
            self.logger.info(correct) # 记录
            self.logger.info(pos_ids) # 记录
            print("Correct: {}/{}".format(correct, len(pos_ids))) # 打印
            if self.replacement:
                self.ids = np.concatenate([self.pos_ids[:len(self.pos_ids) // 2], pos_ids]) # 如果置换的话，在ids的基础上加上neg_ids
            else:
                if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids[:len(self.pos_ids) // 2]]) # 如果为空的话则首先加上pos_ids
                self.ids = np.concatenate([self.ids, pos_ids])
            
            self.ids = self.ids.astype(int) # 为了做差集
            out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
            #assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
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

            self.P[pos_ids] = 1 # 将他们标注为1
            self.logger.info(pcorrect) # 记录
            self.logger.info(ncorrect)
            self.logger.info(pos_ids) # 记录
            self.logger.info(neg_ids)
            print("P Correct: {}/{}".format(pcorrect, len(pos_ids))) # 打印
            print("N Correct: {}/{}".format(ncorrect, len(neg_ids)))

            self.pick_accuracy.append((pcorrect + ncorrect) * 1.0 / (len(pos_ids) * 2))
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


        elif self.mode == 'T':
            self.reset_labels()
            
            al = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)
            print(lt)
            print(ht)
            negative_confident_num = int(lt * len(al))
            positive_confident_num = int((1-ht) * len(al))
            neg_ids = al[:negative_confident_num]
            pos_ids = al[len(al)-positive_confident_num:]

            # 变成向量
            pos_ids = np.array(pos_ids) 
            pos_label = self.T[pos_ids] # 获得neg_ids的真实标签
            pcorrect = np.sum(pos_label == 1) # 抽取N的时候真实标签为-1

            neg_ids = np.array(neg_ids) 
            neg_label = self.T[neg_ids] # 获得neg_ids的真实标签
            ncorrect = np.sum(neg_label < 1)

            self.P[pos_ids] = 1 # 将他们标注为1
            self.logger.info(pcorrect) # 记录
            self.logger.info(ncorrect)
            self.logger.info(pos_ids) # 记录
            self.logger.info(neg_ids)
            print("P Correct: {}/{}".format(pcorrect, len(pos_ids))) # 打印
            print("N Correct: {}/{}".format(ncorrect, len(neg_ids)))

            self.pick_accuracy.append((pcorrect + ncorrect) * 1.0 / (len(pos_ids) * 2))
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
            if self.pickout:
                assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
            return out
        
        
class MNIST_Dataset_FixSample(Dataset):

    def __init__(self, logpath, taskname, n_labeled, n_unlabeled, trainX, trainY, testX, testY, type="noisy", split="train", mode="A", ids=None, pn=False, increasing=False, replacement=True, top = 0.5, flex = 0, pickout=True, seed = None):

        self.X, self.Y, self.T, self.oids, self.prior = make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
        assert np.all(self.oids == np.linspace(0, len(self.X) - 1, len(self.X)))
        self.clean_ids = []
        #self.Y_origin = self.Y
        self.P = self.Y.copy()
        self.type = type
        if (ids is None):
            self.ids = self.oids
        else:
            self.ids = np.array(ids)

        self.split = split
        self.mode = mode
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        logname = logpath + taskname + "_dataset.log"
        if not os.path.exists(logpath):
            os.mkdir(logpath)
        fh = logging.FileHandler(logname, mode="w")
        self.logger.addHandler(fh)
        self.pos_ids = self.oids[self.Y == 1]


        self.pid = self.pos_ids
        if len(self.ids) != 0:
            self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
        else:
            self.uid = []
        print(len(self.uid))
        print(len(self.pid))
        self.sample_ratio = len(self.uid) // len(self.pid)  + 1
        print(self.sample_ratio)
        print("origin:", len(self.pos_ids), len(self.ids))
        self.increasing = increasing
        self.replacement = replacement
        self.top = top
        self.flex = flex
        self.pickout = pickout

        self.pick_accuracy = []
        self.result = -np.ones(len(self))

        self.random_count = 0
    def copy(self, dataset):
        ''' Copy random sequence
        '''
        self.X, self.Y, self.T, self.oids = dataset.X.copy(), dataset.Y.copy(), dataset.T.copy(), dataset.oids.copy()
        self.P = self.Y.copy()
    def __len__(self):
        if self.type == 'noisy':

            #return len(self.uid) * 2
            return len(self.pid) * self.sample_ratio
        else:
            return len(self.ids)

    def set_type(self, type):
        self.type = type

    def update_prob(self, result):
        rank = np.empty_like(result)
        rank[np.argsort(result)] = np.linspace(0, 1, len(result))
        #print(rank)
        if (len(self.pos_ids) > 0):
            rank[self.pos_ids] = -1
        self.result = rank
        
    def shuffle(self):
        perm = np.random.permutation(len(self.uid))
        self.uid = self.uid[perm]

        perm = np.random.permutation(len(self.pid))
        self.pid = self.pid[perm]

    def __getitem__(self, idx): 
        #print(idx)
        # self.ids[idx]是真实的行索引
        # 始终使用真实的行索引去获得数据

        # 1901 保持比例
        if self.type == 'noisy':
            '''
            if (idx % 2 == 0):
                index = self.pid[idx % 1000]
            else:
                index = self.uid[idx - (idx // 2 + 1)]

            '''
            if (idx % self.sample_ratio == 0):
                index = self.pid[idx // self.sample_ratio]
                id = 0
            else:
                index = self.uid[idx - (idx // self.sample_ratio + 1)]
                
            return self.X[index], self.Y[index], self.P[index], self.T[index], index, 0
        else:
            return self.X[self.ids[idx]], self.Y[self.ids[idx]], self.P[self.ids[idx]], self.T[self.ids[idx]], self.ids[idx], 0


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
            if len(self.pid) == 0:
                self.sample_ratio = 10000000000
            else:
                self.sample_ratio = int(len(self.uid) / len(self.pid)) + 1
    def reset_labels(self):
        ''' Reset Y labels
        '''
        self.P = self.Y.copy()

    def update_ids(self, results, epoch, ratio=None, lt = 0, ht = 0):
        if not self.replacement or self.increasing:
            percent = min(epoch / 100, 1) # 决定抽取数据的比例
        else:
            percent = 1
        if ratio == None:
            ratio = self.prior
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

        self.P[pos_ids] = 1 # 将他们标注为1
        self.logger.info(pcorrect) # 记录
        self.logger.info(ncorrect)
        self.logger.info(pos_ids) # 记录
        self.logger.info(neg_ids)
        print("P Correct: {}/{}".format(pcorrect, len(pos_ids))) # 打印
        print("N Correct: {}/{}".format(ncorrect, len(neg_ids)))

        self.pick_accuracy.append((pcorrect + ncorrect) * 1.0 / (len(pos_ids) * 2))
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






