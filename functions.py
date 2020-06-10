import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np


class stats:
    def __init__(self, path, start_epoch):
        if start_epoch is not 0:
            stats_ = sio.loadmat(os.path.join(path,'stats.mat'))
            data = stats_['data']
            content = data[0,0]
            #self.trainObj = content['trainObj'][:,:start_epoch].squeeze().tolist()
            self.trainPacc = content['trainPacc'][:,:start_epoch].squeeze().tolist()
            self.trainNacc = content['trainNacc'][:,:start_epoch].squeeze().tolist()
            self.trainPNacc = content['trainPNacc'][:,:start_epoch].squeeze().tolist()
            #self.valObj = content['valObj'][:,:start_epoch].squeeze().tolist()
            self.valPacc = content['valPacc'][:,:start_epoch].squeeze().tolist()
            self.valNacc = content['valNacc'][:,:start_epoch].squeeze().tolist()
            self.valPNacc = content['valPNacc'][:,:start_epoch].squeeze().tolist()
            if start_epoch is 1:
                #self.trainObj = [self.trainObj]
                self.trainPacc = [self.trainPacc]
                self.trainNacc = [self.trainNacc]
                self.trainPNacc = [self.trainPNacc]
                #self.valObj = [self.valObj]
                self.valPacc = [self.valPacc]
                self.valNacc = [self.valNacc]
                self.valPNacc = [self.valPNacc]
        else:
            #self.trainObj = []
            self.trainPacc = []
            self.trainNacc = []
            self.trainPNacc = []
            #self.valObj = []
            self.valPacc = []
            self.valNacc = []
            self.valPNacc = []
    def _update(self, trainPacc, trainNacc, trainPNacc, valPacc, valNacc, valPNacc):
        #self.trainObj.append(trainObj)
        self.trainPacc.append(trainPacc.cpu().numpy())
        self.trainNacc.append(trainNacc.cpu().numpy())
        self.trainPNacc.append(trainPNacc.cpu().numpy())
        #self.valObj.append(valObj)
        self.valPacc.append(valPacc.cpu().numpy())
        self.valNacc.append(valNacc.cpu().numpy())
        self.valPNacc.append(valPNacc.cpu().numpy())


def plot_curve(stats, path, taskname, iserr):
    #trainObj = np.array(stats.trainObj)
    #valObj = np.array(stats.valObj)
    if iserr:
        trainPNacc = 100 - np.array(stats.trainPNacc)
        valPNacc = 100 - np.array(stats.valPNacc)
        trainPacc = 100 - np.array(stats.trainPacc)
        valPacc = 100 - np.array(stats.valPacc)
        trainNacc = 100 - np.array(stats.trainNacc)
        valNacc = 100 - np.array(stats.valNacc)
        titleName = 'error'
    else:
        trainPNacc = np.array(stats.trainPNacc)
        valPNacc = np.array(stats.valPNacc)
        trainPacc = np.array(stats.trainPacc)
        valPacc = np.array(stats.valPacc)
        trainNacc = np.array(stats.trainNacc)
        valNacc = np.array(stats.valNacc)
        titleName = 'accuracy'
    epoch = len(trainPacc)
    figure = plt.figure()
    #obj = plt.subplot(2,2,1)
    #obj.plot(range(1,epoch+1),trainObj,'o-',label = 'train')
    #obj.plot(range(1,epoch+1),valObj,'o-',label = 'val')
    #plt.xlabel('epoch')
    #plt.title('objective')
    #handles, labels = obj.get_legend_handles_labels()
    #obj.legend(handles[::-1], labels[::-1])
    top1 = plt.subplot(3,1,1)
    top1.plot(range(1,epoch+1),trainPacc,'o-',label = 'trainP')
    top1.plot(range(1,epoch+1),valPacc,'o-',label = 'valP')
    plt.title('top1 P ' + titleName)
    plt.xlabel('epoch')
    handles, labels = top1.get_legend_handles_labels()
    top1.legend(handles[::-1], labels[::-1])

    top1 = plt.subplot(3,1,2)
    top1.plot(range(1,epoch+1),trainNacc,'o-',label = 'trainN')
    top1.plot(range(1,epoch+1),valNacc,'o-',label = 'valN')
    plt.title('top1 N ' + titleName)
    plt.xlabel('epoch')
    handles, labels = top1.get_legend_handles_labels()
    top1.legend(handles[::-1], labels[::-1])

    top1 = plt.subplot(3,1,3)
    top1.plot(range(1,epoch+1),trainPNacc,'o-',label = 'trainPN')
    top1.plot(range(1,epoch+1),valPNacc,'o-',label = 'valPN')
    plt.title('top1 PN ' + titleName)
    plt.xlabel('epoch')
    handles, labels = top1.get_legend_handles_labels()
    top1.legend(handles[::-1], labels[::-1])


    filename = os.path.join(path, taskname + '_net-train.pdf')
    figure.savefig(filename, bbox_inches='tight')
    plt.close()

def decode_params(input_params):
    params = input_params[0]
    out_params = []
    _start=0
    _end=0
    for i in range(len(params)):
        if params[i] == ',':
            out_params.append(float(params[_start:_end]))
            _start=_end+1
        _end+=1
    out_params.append(float(params[_start:_end]))
    return out_params
