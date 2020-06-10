import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from skimage import transform
import os
from random import choice
from scipy.ndimage.filters import gaussian_filter

class ADNI(Dataset):
    """ADNI Dataset."""

    def __init__(self, csv_file, nids, ids, data_path, type = 'noisy', transform = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.metadata = self.metadata.fillna('')
        self.nids = nids
        #print(nids)
        self.ids = ids
        self.oids = np.array(list(range(len(self.nids))))
        self.prior = 0.5
        self.flex = 0
        if (ids is None):
            self.ids = self.oids
        else:
            self.ids = np.array(ids)
        self.data_path = data_path
        self.split=1
        self.size=(50, 50, 50)
        self.transform = transform

        self.type    = type
        #print(self.oids)
        #print(len(self.oids))
        self.Y = np.zeros(len(self.oids))
        self.T = np.zeros(len(self.oids))
        self.prefetch()
        self.Y = self.Y.astype(int)
        self.T = self.T.astype(int)
        print(np.sum(self.Y == 1))
        print(np.sum(self.Y == -1))
        print(np.sum(self.T == 1))
        print(np.sum(self.T == -1))
        print(self.oids)
        self.P = self.Y.copy()
        print(self.Y.dtype)
        self.pos_ids = self.oids[self.Y == 1]
        self.pid = self.pos_ids

        if len(self.ids) != 0:
            self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
        else:
            self.uid = []

        self.sample_ratio = len(self.uid) // len(self.pid)  + 1
        print("origin:", len(self.pos_ids), len(self.ids))
        self.increasing = True
        self.replacement = True
        self.top = 0.5
        self.pickout = True

    def set_type(self, type):
        self.type = type
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

    def shuffle(self):
        perm = np.random.permutation(len(self.uid))
        self.uid = self.uid[perm]

        perm = np.random.permutation(len(self.pid))
        self.pid = self.pid[perm]

    def _transform_shift(self, image):
        # https://pytorch.org/docs/stable/torchvision/transforms.html
        # https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/6
        shift_range = list(range(-2, 3))
        shift_x = choice(shift_range)
        shift_y = choice(shift_range)
        shift_z = choice(shift_range)
        image = np.roll(image, shift_x, 0)
        if shift_x >= 0: image[:shift_x, :, :] = 0
        else: image[shift_x:, :, :] = 0
        image = np.roll(image, shift_y, 1)
        if shift_y >= 0: image[:, :shift_y, :] = 0
        else: image[:, shift_y:, :] = 0
        image = np.roll(image, shift_z, 2)
        if shift_z >= 0: image[:, :, :shift_z] = 0
        else: image[:, :, shift_z:] = 0
        
        return image
    
    def _transform_gaussian(self, image, sigma=None):

        if not sigma: sigma = np.random.rand() * 0.5
        image = gaussian_filter(image * 1.0, sigma=sigma)
        
        return image
    
    def _transform_sagittal_flip(self, image):
        image = np.flip(image, 0)
        return image
    
    def _transform_noise(self, image):
        shape = image.shape
        image = image + np.random.randn(*shape).astype('float32')
        return image

    def _transform(self, image):
        if np.random.random() > 0.5:
            image = self._transform_sagittal_flip(image).copy()

        return image
    
    def chunk2channel(self, image, split=2):
        '''image: h*w*d'''
        shape = image.shape
        chunks_x = np.split(image, split, axis=0)
        chunks_xy = []
        for chunk in chunks_x:
            chunks_xy += np.split(chunk, split, axis=1)
        chunks_xyz = []
        for chunk in chunks_xy:
            chunks_xyz += np.split(chunk, split, axis=2)
        chunks = np.zeros((split**3, shape[0]//split, shape[1]//split, shape[2]//split)).astype('float32')
        for i in range(len(chunks_xyz)):
            chunks[i] = chunks_xyz[i]
        return chunks
    
    def get_image(self, id, name):
        ''' name: mri/grey/white/pet '''
        '''[8:112, 10:138, :112] OR [8:113, 10:139, :114]'''
        if self.split == 3:
            image = np.load(os.path.join(self.data_path, name, id + "." + name + ".npy"))[8:113, 10:139, :114]
        else:
            image = np.load(os.path.join(self.data_path, name, id + "." + name + ".npy"))[8:112, 10:138, :112]
        image = image.astype('float32')
        if self.transform:
            image = self._transform(image)
        image /= image.max()
        if self.split > 1: image = self.chunk2channel(image, split=self.split)
        return image
    
    def get_multi_modal(self, id, name):
        image = np.load(os.path.join(self.data_path, name, id + "." + name + ".npy"))[8:112, 10:138, :112]
        image /= image.max()
        image = image.astype('float32')
        
        transformed = self._transform(image)
        cropped = image[24:74, 49:75, 24:70]
        if self.split > 1:
            image = self.chunk2channel(image, self.split)
            transformed = self.chunk2channel(transformed, self.split)
            cropped = self.chunk2channel(cropped, self.split)
        else:
            image = np.expand_dims(image, axis=0) # add one channel dimension: (1, 104, 128, 112)
            transformed = np.expand_dims(transformed, axis=0) # add one channel dimension: (1, 104, 128, 112)
        return image, transformed, cropped

    def get_hippo(self, image):
        # left hippocampus as [64:90, 59:85, 24:50] and right hippocampus as [31:57, 59:85, 24:50] in [121, 145, 121]
        # 30,30,30 => [54:84, 47:77, 22:52] and [21:51, 47:77, 22:52] in [104, 128, 112] ([8:112, 10:138, :112])
        left = image[54:84, 47:77, 22:52]
        right = image[21:51, 47:77, 22:52]
        return left, right

    def prefetch(self):
        for i in range(len(self.nids)):
            id = self.nids[i]
            rid = int(id.split('.')[0])
            if '.' in id: image_id = int(id.split('.')[1])
            else: image_id = ''
            suvr = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['SUVR'].values.astype('float32')
            if suvr >= 1.18:
                self.Y[i] = 1
                self.T[i] = 1
            elif suvr >= 1.08:
                self.Y[i] = -1
                self.T[i] = 1
            else:
                self.Y[i] = -1
                self.T[i] = -1

    def __getitem__(self, idx): 
        # self.ids[idx]是真实的行索引
        # 始终使用真实的行索引去获得数据

        # 1901 保持比例
        if self.type == 'noisy':
            if (idx % self.sample_ratio == 0):
                trueid = self.pid[idx // self.sample_ratio]
                id = self.nids[trueid]
                X = self.get_image(id, 'mri')
                Y = self.Y[trueid]
                P = self.P[trueid]
                T = self.T[trueid]
                left, right = self.get_hippo(X)
                left = np.expand_dims(left, axis=0).astype('float32')
                right = np.expand_dims(right, axis=0).astype('float32')
                X = transform.resize(X, self.size)
                X = np.expand_dims(X, axis=0).astype('float32')
                return X, left, right, Y, P, T, self.ids[idx // self.sample_ratio]
            else:
                trueid = self.uid[idx - (idx // self.sample_ratio + 1)]
                id = self.nids[trueid]
                X = self.get_image(id, 'mri')
                Y = self.Y[trueid]
                P = self.P[trueid]
                T = self.T[trueid]
                left, right = self.get_hippo(X)
                left = np.expand_dims(left, axis=0).astype('float32')
                right = np.expand_dims(right, axis=0).astype('float32')
                X = transform.resize(X, self.size)
                X = np.expand_dims(X, axis=0).astype('float32')
                return X, left, right, Y, P, T, self.ids[idx - (idx // self.sample_ratio + 1)]
        else:
            trueid = self.ids[idx]
            id = self.nids[trueid]
            X = self.get_image(id, 'mri')
            Y = self.Y[trueid]
            P = self.P[trueid]
            T = self.T[trueid]
            left, right = self.get_hippo(X)
            left = np.expand_dims(left, axis=0).astype('float32')
            right = np.expand_dims(right, axis=0).astype('float32')
            X = transform.resize(X, self.size)
            X = np.expand_dims(X, axis=0).astype('float32')
            #print(X.shape)
            return X, left, right, Y, P, T, self.ids[idx]
    '''
    def __getitem__(self, id):
        if self.mode == 'noisy':
        id = self.ids[id]
        sample = {'id': id}
        
        if self.mri:
            sample['mri'] = self.get_image(id, 'mri')
            if self.noise:
                sample['mri_noise'] = self._transform_noise(sample['mri'])
            if self.hippo:
                sample['left'], sample['right'] = self.get_hippo(sample['mri'])
                sample['left'] = np.expand_dims(sample['left'], axis=0).astype('float32')
                sample['right'] = np.expand_dims(sample['right'], axis=0).astype('float32')
            if self.size is not None:
                sample['mri'] = transform.resize(sample['mri'], self.size)
            sample['mri'] = np.expand_dims(sample['mri'], axis=0).astype('float32')
            
        if self.grey:
            if self.multi_modal:
                image, transformed, cropped = self.get_multi_modal(id, 'grey')
                sample['grey'] = image
                sample['grey_transform'] = transformed
                sample['grey_hippo'] = cropped
            else:
                sample['grey'] = self.get_image(id, 'grey')
                if self.transform:
                    sample['grey'] = self._transform(sample['grey'])
                if self.noise:
                    sample['grey_noise'] = self._transform_noise(sample['grey'])
            
        if self.white:
            sample['white'] = self.get_image(id, 'white')
            if self.transform:
                sample['white'] = self._transform(sample['white'])
            
        if self.csf:
            sample['csf'] = self.get_image(id, 'csf')
            if self.transform:
                sample['csf'] = self._transform(sample['csf'])
            
        if self.pet:
            sample['pet'] = self.get_image(id, 'pet')
        

        rid = int(id.split('.')[0])
        if '.' in id: image_id = int(id.split('.')[1])
        else: image_id = ''

        if self.suvr:
            sample['suvr'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['SUVR'].values.astype('float32')
            
        if self.dx:
            # 1=NL 2=MCI, 3=AD
            sample['dx'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['DX'].values.astype('float32')
            sample['dx'] -= 1
            # for NL(0)/AD(1) clf
            if sample['dx'][0] == 2: sample['dx'][0] = 1

        if self.age:
            sample['age'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['Age'].values.astype('float32')
            sample['age'] /= 100.
        
        if self.gender:
            # 1=male, 2=female
            sample['gender'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['Gender'].values.astype('float32')
            sample['gender'] -= 1

        if self.edu:
            sample['edu'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['Education'].values.astype('float32')
            sample['edu'] /= 25.

        if self.apoe:
            sample['apoe'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['ApoE4'].values[0]
            sample['apoe'] = np.array(self.apoe_dict[sample['apoe']]).reshape(1).astype('float32')
            sample['apoe'] /= 3
        
        return sample
    '''
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

    def update_ids(self, results, epoch, ratio=None):
        
        percent = min(epoch / 100, 1) # 决定抽取数据的比例

        ratio = self.prior

        self.reset_labels()
        n_all = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent * self.top) # 决定抽取的数量
        print(len(self.oids))
        print(len(self.pos_ids))
        print(ratio)
        print(n_all)
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
        #self.logger.info(pcorrect) # 记录
        #self.logger.info(ncorrect)
        #self.logger.info(pos_ids) # 记录
        #self.logger.info(neg_ids)
        print("P Correct: {}/{}".format(pcorrect, len(pos_ids))) # 打印
        print("N Correct: {}/{}".format(ncorrect, len(neg_ids)))
            #self.ids = np.concatenate([self.pos_ids, pos_ids, neg_ids]) # 如果置换的话，在ids的基础上加上neg_ids
        self.ids = np.concatenate([pos_ids, neg_ids]) 
        
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
