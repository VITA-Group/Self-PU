import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def parameters(self):
       for name, param in self.named_params(self):
            yield param
    
    def named_leaves(self):
        return []
    
    def named_submodules(self):
        return []
    
    def named_params(self, curr_module=None, memo=None, prefix=''):       
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
                    
        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p
    
    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is None:
                    print(name_t)
                    tmp = param_t
                else:
                    tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    if grad is None:
                        print(name)
                        tmp = param_t
                    else:
                        tmp = param_t - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self,curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)
            
    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())   
                
    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
       
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        if ignore.bias is not None: self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else: self.bias = None
    def forward(self, x):
        if self.bias is not None:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight)
    
    def named_leaves(self):
        if self.bias is not None:
            return [('weight', self.weight), ('bias', self.bias)]
        else:
            return [('weight', self.weight)]
    
class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaConv3d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv3d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x):
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
       
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaBatchNorm3d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm3d(*args, **kwargs)
        
        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:           
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

        
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)
            
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)
        
        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:           
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

        
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)
            
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)
        
        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats
       
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

        
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)
        
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
        

class LeNet(MetaModule):
    def __init__(self, n_out):
        super(LeNet, self).__init__()
    
        layers = []
        layers.append(MetaConv2d(1, 6, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(MetaConv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        
        layers.append(MetaConv2d(16, 120, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        
        self.main = nn.Sequential(*layers)
        
        layers = []
        layers.append(MetaLinear(120, 84))
        layers.append(nn.ReLU(inplace=True))
        layers.append(MetaLinear(84, n_out))
        
        self.fc_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        return self.fc_layers(x).squeeze()
    
class MetaMLP(MetaModule):
    def __init__(self, dim):
        super(MetaMLP, self).__init__()
        layers = []
        self.l1 = MetaLinear(dim, 300, bias=False)
        self.bn1 = MetaBatchNorm1d(num_features = 300)
        self.l2 = MetaLinear(300, 300, bias=False)
        self.bn2 = MetaBatchNorm1d(num_features = 300)
        self.l3 = MetaLinear(300, 300, bias=False)
        self.bn3 = MetaBatchNorm1d(num_features = 300)
        self.l4 = MetaLinear(300, 300, bias=False)
        self.bn4 = MetaBatchNorm1d(num_features = 300)
        self.l5 = MetaLinear(300, 1)
    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.l4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = x.view(-1, 300)
        x = self.l5(x)
        return x
    
class MetaCNN(MetaModule):

    def __init__(self):
        super(MetaCNN, self).__init__()
        self.conv1 = MetaConv2d(3, 96, kernel_size = 3, padding=1)
        self.bn1 = MetaBatchNorm2d(96)
        self.conv2 = MetaConv2d(96, 96, kernel_size = 3, padding=1)
        self.bn2 = MetaBatchNorm2d(96)
        self.conv3 = MetaConv2d(96, 96, kernel_size = 3, stride = 2, padding=1)
        self.bn3 = MetaBatchNorm2d(96)
        self.conv4 = MetaConv2d(96, 192, kernel_size = 3, padding=1)
        self.bn4 = MetaBatchNorm2d(192)
        self.conv5 = MetaConv2d(192, 192, kernel_size = 3, padding=1)
        self.bn5 = MetaBatchNorm2d(192)
        self.conv6 = MetaConv2d(192, 192, kernel_size = 3, stride = 2, padding=1)
        self.bn6 = MetaBatchNorm2d(192)
        self.conv7 = MetaConv2d(192, 192, kernel_size = 3, padding=1)
        self.bn7 = MetaBatchNorm2d(192)
        self.conv8 = MetaConv2d(192, 192, kernel_size = 1)
        self.bn8 = MetaBatchNorm2d(192)
        self.conv9 = MetaConv2d(192, 10, kernel_size = 1)
        self.bn9 = MetaBatchNorm2d(10)
        self.l1 = MetaLinear(640, 1000)
        self.l2 = MetaLinear(1000, 1000)
        self.l3 = MetaLinear(1000, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = x.view(-1, 640)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return x
    
    
class MetaBasic2Conv(MetaModule):
    def __init__(self):
        super(MetaBasic2Conv, self).__init__()

        self.conv1 = MetaConv3d(1, 16, kernel_size=3, stride=1, padding=1) # b, 16, 61, 73, 61
        self.bn1 = MetaBatchNorm3d(16)
        self.conv2 = MetaConv3d(16, 32, kernel_size=3, stride=1, padding=1)  # b, 32, 31, 37, 31
        self.bn2 = MetaBatchNorm3d(32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace = True)
        x = F.max_pool3d(x, kernel_size = 3, stride = 2, padding = 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace = True)
        x = F.max_pool3d(x, kernel_size = 3, stride = 2, padding = 1)
        return x



class MetaLenet3D(MetaModule):
    def __init__(self):
        super(MetaLenet3D, self).__init__()
        self.conv_mri = MetaBasic2Conv().cuda()
        self.conv_left = MetaBasic2Conv().cuda()
        self.conv_right = MetaBasic2Conv().cuda()
        self.fc1 = MetaLinear(32 * (13*13*13 + 2*8*8*8), 256)
        self.bn1 = MetaBatchNorm1d(256)
        self.fc2 = MetaLinear(256, 1)


    def forward(self, mri, left, right):
        mri = self.conv_mri(mri)
        left = self.conv_left(left)
        right = self.conv_right(right)
        # print(mri.size(), left.size(), right.size())
        mri = mri.view(-1, 32 * 13 * 13 * 13)
        left = left.view(-1, 32 * 8 * 8 * 8)
        right = right.view(-1, 32 * 8 * 8 * 8)
        x = torch.cat((mri, left, right), dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace = True)
        x = self.fc2(x)
        #x = F.softmax(x, 1)
        #x = torch.log(x[:, 0] / (1- x[: ,0])).view(-1, 1)
        return x
