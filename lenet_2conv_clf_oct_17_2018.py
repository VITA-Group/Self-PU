import torch
from torch import nn
import torch.nn.functional as F

'''
AD/NL classification
'''
class Basic2Conv(nn.Module):
    def __init__(self):
        super(Basic2Conv, self).__init__()
        self.conv = nn.Sequential(
            # 121, 145, 121
            # padding tuple (padT, padH, padW)
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1), # b, 16, 61, 73, 61
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # b, 16, 31, 37, 31
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),  # b, 32, 31, 37, 31
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # b, 32, 16, 19, 16
        )
        nn.init.xavier_uniform(self.conv[0].weight)
        nn.init.xavier_uniform(self.conv[4].weight)

    def forward(self, x):
        return self.conv(x)



class Lenet3D(nn.Module):
    def __init__(self):
        super(Lenet3D, self).__init__()
        self.conv_mri = Basic2Conv().cuda()
        self.conv_left = Basic2Conv().cuda()
        self.conv_right = Basic2Conv().cuda()
        self.fc = nn.Sequential(
            nn.Linear(32 * (13*13*13 + 2*8*8*8), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            #nn.ReLU(True)
        )
        nn.init.xavier_uniform(self.fc[0].weight)

    def forward(self, mri, left, right):
        mri = self.conv_mri(mri)
        left = self.conv_left(left)
        right = self.conv_right(right)
        # print(mri.size(), left.size(), right.size())
        mri = mri.view(-1, 32 * 13 * 13 * 13)
        left = left.view(-1, 32 * 8 * 8 * 8)
        right = right.view(-1, 32 * 8 * 8 * 8)
        x = torch.cat((mri, left, right), dim=1)
        x = self.fc(x)
        #x = F.softmax(x, 1)
        #x = torch.log(x[:, 0] / (1- x[: ,0])).view(-1, 1)
        return x
