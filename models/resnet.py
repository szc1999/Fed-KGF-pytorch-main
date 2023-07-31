'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.layers.shift import ShiftConv2d
from models.layers.ghostconv import GhostConv2d
from models.layers.lbconv import LocalBinaryConv2d
from models.layers.perturbconv import PerturbativeConv2d
from models.layers.monoconv import ShieldMonomialConv2d, ShieldRBFFamilyConv2d


__all__ = [
    'resnet18', 'monomial_resnet18', 'gaussian_resnet18', 'multiquadric_resnet18',
    'inverse_quadratic_resnet18', 'inverse_multiquadric_resnet18',
    'local_binary_resnet18', 'perturbative_resnet18', 'ghost_resnet18',
    'resnet34', 'monomial_resnet34', 'local_binary_resnet34', 'perturbative_resnet34',
    'ghost_resnet34'
]

# Block
class BegeinBlock(nn.Module):#ResNet网络中的基本模块
    def __init__(self, in_planes, planes, stride=1):
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out 

class BasicBlock(nn.Module):#ResNet网络中的基本模块
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:#如果条件达成，则添加一个用于调整维度的卷积层和批归一化层，构成了残差连接（shortcut）
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)# 残差连接
        out = F.relu(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MonomialBasicBlock(nn.Module):
    expansion = 1
    '''
    Block结构：
    1.Conv2d or MonoConv2d
    2.BatchNorm2d

    3.MonoConv2d
    4.BatchNorm2d

    5.shortcut
    '''

    def __init__(self, in_planes, planes, stride=2, num_terms=3, exp_range=(1, 10), exp_factor=1, mono_bias=False,
                 onebyone=False, ):#quilk=False
        super(MonomialBasicBlock, self).__init__()

        # self.quilk = quilk
        self.stride = stride
        if self.stride > 1:#如果 stride > 1，则使用普通的卷积层，否则使用单项式卷积层（#Pmonocnn）
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        else:
            self.conv1 = ShieldMonomialConv2d(
                in_planes, planes, num_seeds=planes // exp_factor, num_terms=num_terms,
                exp_range=exp_range, fanout_factor=exp_factor, kernel_size=3, stride=1, padding=1,
                mono_bias=mono_bias, onebyone=onebyone
            )
            # self.conv1_q = ShieldMonomialConv2d(
            #     in_planes, planes, num_seeds=planes // int(exp_factor / 4), num_terms=num_terms,
            #     exp_range=exp_range, fanout_factor=int(exp_factor/4), kernel_size=3, stride=1, padding=1,
            #     mono_bias=mono_bias, onebyone=onebyone
            # )

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ShieldMonomialConv2d(
            planes, planes, num_seeds=planes // exp_factor, num_terms=num_terms,
            exp_range=exp_range, fanout_factor=exp_factor, kernel_size=3, stride=1, padding=1,
            mono_bias=mono_bias, onebyone=onebyone
        )
        # self.conv2_q = ShieldMonomialConv2d(
        #     planes, planes, num_seeds=planes // int(exp_factor / 4), num_terms=num_terms,
        #     exp_range=exp_range, fanout_factor=int(exp_factor/4), kernel_size=3, stride=1, padding=1,
        #     mono_bias=mono_bias, onebyone=onebyone
        # )

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:#如果条件达成，则添加一个用于调整维度的卷积层和批归一化层，构成了残差连接（shortcut）
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # if self.quilk and self.stride == 1:
        #     out = F.relu(self.bn1(self.conv1_q(x)))
        #     out = self.bn2(self.conv2_q(out))
        # else:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RBFFamilyBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, exp_range=(1, 10), exp_factor=2, rbf='gaussian', onebyone=False):
        super(RBFFamilyBasicBlock, self).__init__()

        if stride > 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = ShieldRBFFamilyConv2d(
                in_planes, planes, num_seeds=planes // exp_factor, eps_range=exp_range, fanout_factor=exp_factor,
                kernel_size=3, stride=1, padding=1, bias=False, rbf=rbf, onebyone=onebyone)

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ShieldRBFFamilyConv2d(
            planes, planes, num_seeds=planes // exp_factor, eps_range=exp_range, fanout_factor=exp_factor,
            kernel_size=3, stride=1, padding=1, bias=False, rbf=rbf, onebyone=onebyone)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LocalBinaryBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sparsity=0.1):
        super(LocalBinaryBasicBlock, self).__init__()

        if stride > 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = LocalBinaryConv2d(in_planes, planes, sparsity, kernel_size=3, stride=1, padding=1, bias=False,
                                           mid_channels=self.expansion*planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = LocalBinaryConv2d(planes, planes, sparsity, kernel_size=3, stride=1, padding=1, bias=False,
                                       mid_channels=self.expansion*planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GhostBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, ratio=4):
        super(GhostBasicBlock, self).__init__()

        if stride > 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = GhostConv2d(in_planes, planes, ratio=ratio, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = GhostConv2d(planes, planes, ratio=ratio, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PerturbationBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, noise_level=0.1, noisy_train=False, noisy_eval=False):
        super(PerturbationBasicBlock, self).__init__()

        if stride > 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = PerturbativeConv2d(
                in_planes, planes, noise_level, noisy_train=noisy_train, noisy_eval=noisy_eval)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = PerturbativeConv2d(
            planes, planes, noise_level, noisy_train=noisy_train, noisy_eval=noisy_eval)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# class ShiftBasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(ShiftBasicBlock, self).__init__()

#         if stride > 1:
#             self.conv1 = nn.Conv2d(
#                 in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         else:
#             self.conv1 = ShiftConv2d(in_planes, planes)
#         self.bn1 = nn.BatchNorm2d(planes)

#         self.conv2 = ShiftConv2d(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# Net
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, num_classes=10, features=False, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.features = features
        self.layer0_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer0_bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, **kwargs)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.layer0_bn1(self.layer0_conv1(x)))

        if self.features and self.training:
            feats = []
            out = self.layer1(out)
            feats.append(out)
            out = self.layer2(out)
            feats.append(out)
            out = self.layer3(out)
            feats.append(out)
            out = self.layer4(out)
            feats.append(out)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.features and self.training:
            return out, feats
        else:
            return out


class MonomialResNet(nn.Module):# 与class ResNet区别不同于block不一样
    def __init__(self,
                 block: MonomialBasicBlock,
                 num_blocks, num_terms, exp_range, exp_factor, mono_bias, onebyone, 
                 in_channels=3, num_classes=10, features=True):
        super(MonomialResNet, self).__init__()
        '''
            block：表示要使用的基本块类，这里使用的是 MonomialBasicBlock。
            num_blocks：一个包含四个整数的列表，表示每个阶段中基本块的数量。
            num_terms：表示单项式卷积中单项式的数量。
            exp_range：一个包含两个整数的元组，表示单项式卷积中单项式指数的范围。
            exp_factor：一个包含四个整数的列表，表示每个阶段中单项式卷积的扩展因子。
            mono_bias：一个布尔值，表示单项式卷积是否使用偏置。
            onebyone：一个布尔值，表示单项式卷积是否使用1x1卷积
        '''

        self.in_planes = 64 #跟踪当前层的输入通道数，并通过乘以block.expansion获得输出通道数
        self.features = features #是否输出特征
        self.fuse = False #是否融合特征
        self.feats_y = []
        
        self.layer0_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer0_bn1 = nn.BatchNorm2d(64)
        # self.fuse_conv0 = nn.Conv2d(64 * 2, 64, kernel_size=1)
        # self.fuse_conv1 = nn.Conv2d(128 * 2, 128, kernel_size=1)
        # self.fuse_conv2 = nn.Conv2d(256 * 2, 256, kernel_size=1)
        # self.fuse_conv3 = nn.Conv2d(512 * 2, 512, kernel_size=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, num_terms, exp_range, exp_factor[0], mono_bias,
                                       onebyone)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, num_terms, exp_range, exp_factor[1], mono_bias,
                                       onebyone)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, num_terms, exp_range, exp_factor[2], mono_bias,
                                       onebyone)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, num_terms, exp_range, exp_factor[3], mono_bias,
                                       onebyone)

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self,
                    block: MonomialBasicBlock,
                    planes, num_blocks, stride, num_terms, exp_range, exp_factor, mono_bias, onebyone):
        strides = [stride] + [1]*(num_blocks-1)# 各block的步长[stride,1,1...1,1]
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride, num_terms, exp_range, exp_factor, mono_bias, onebyone))#, quilk=False
            self.in_planes = planes * block.expansion#基本块都会接收in_planes作为输入通道数，并根据planes*block.expansion计算出输出通道数。然后，self.in_planes的值会更新为输出通道数
        return nn.Sequential(*layers)

    # def quilk(self, name_layer):
    #     idx_layer = name_layer[5]
    #     idx_block = name_layer[7]
    #     if idx_layer == '1':
    #         self.layer1[int(idx_block)].quilk = True
    #     elif idx_layer == '2':
    #         self.layer2[int(idx_block)].quilk = True
    #     elif idx_layer == '3':
    #         self.layer3[int(idx_block)].quilk = True
    #     elif idx_layer == '4':
    #         self.layer4[int(idx_block)].quilk = True
    
    # def unquilk(self, name_layer):
    #     idx_layer = name_layer[5]
    #     idx_block = name_layer[7]
    #     if idx_layer == '1':
    #         self.layer1[int(idx_block)].quilk = False
    #     elif idx_layer == '2':
    #         self.layer2[int(idx_block)].quilk = False
    #     elif idx_layer == '3':
    #         self.layer3[int(idx_block)].quilk = False
    #     elif idx_layer == '4':
    #         self.layer4[int(idx_block)].quilk = False

    def forward(self, x):
        out = F.relu(self.layer0_bn1(self.layer0_conv1(x)))
        if self.features and self.training:#如果 self.features 为 True 且网络处于训练模式，将当前阶段的特征保存到 feats 列表中
            feats = []
            out = self.layer1(out)
            # if self.fuse: 
            #     # out = self.fuse_conv0(torch.cat((out, self.feats_y[0]), dim=1))
            #     out = self.gamma0 * self.feats_y[0] + (1 - self.gamma0) * out
            feats.append(out)#torch.Size([batch_size, 64, 32, 32])
            out = self.layer2(out)
            # if self.fuse: 
            #     # out = self.fuse_conv1(torch.cat((out, self.feats_y[1]), dim=1))
            #     out = self.gamma1 * self.feats_y[1] + (1 - self.gamma1) * out
            feats.append(out)#torch.Size([batch_size, 128, 16, 16])
            out = self.layer3(out)
            # if self.fuse: 
            #     # out = self.fuse_conv2(torch.cat((out, self.feats_y[2]), dim=1))
            #     out = self.gamma2 * self.feats_y[2] + (1 - self.gamma2) * out
            feats.append(out)#torch.Size([batch_size, 256, 8, 8])
            out = self.layer4(out)
            # if self.fuse: 
            #     # out = self.fuse_conv3(torch.cat((out, self.feats_y[3]), dim=1))
            #     out = self.gamma3 * self.feats_y[3] + (1 - self.gamma3) * out
            feats.append(out)#torch.Size([batch_size, 512, 4, 4])
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

        #经过全局平均池化层、展平操作和全连接层，得到最终的输出
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.features and self.training:
            return out, feats
        else:
            return out


class RBFFamilyResNet(nn.Module):
    def __init__(self, block: RBFFamilyBasicBlock,
                 num_blocks, eps_range, exp_factor, rbf,
                 onebyone=True, in_channels=3, num_classes=10):
        super(RBFFamilyResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, eps_range, exp_factor[0], rbf,
                                       onebyone)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, eps_range, exp_factor[1], rbf,
                                       onebyone)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, eps_range, exp_factor[2], rbf,
                                       onebyone)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, eps_range, exp_factor[3], rbf,
                                       onebyone)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, eps_range, exp_factor, rbf,
                    onebyone):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride, eps_range, exp_factor, rbf, onebyone))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class LocalBinaryResNet(nn.Module):
    def __init__(self, block: LocalBinaryBasicBlock,
                 num_blocks, in_channels=3, num_classes=10, sparsity=0.1, features=True):
        super(LocalBinaryResNet, self).__init__()
        self.in_planes = 64
        self.features = features

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, sparsity)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, sparsity)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, sparsity)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, sparsity)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, sparsity):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sparsity=sparsity))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.features and self.training:
            feats = []
            out = self.layer2(out)
            feats.append(out)
            out = self.layer3(out)
            feats.append(out)
            out = self.layer4(out)
            feats.append(out)
        else:
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.features and self.training:
            return out, feats
        else:
            return out


class PerturbationResNet(nn.Module):
    def __init__(self, block: PerturbationBasicBlock,
                 num_blocks, noise_level, noisy_train, noisy_eval, in_channels=3, num_classes=10):
        super(PerturbationResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, noise_level, noisy_train, noisy_eval)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, noise_level, noisy_train, noisy_eval)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, noise_level, noisy_train, noisy_eval)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, noise_level, noisy_train, noisy_eval)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block: PerturbationBasicBlock,
                    planes, num_blocks, stride, noise_level, noisy_train, noisy_eval):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                noise_level=noise_level, noisy_train=noisy_train, noisy_eval=noisy_eval))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


# class ShiftResNet(nn.Module):
#     def __init__(self, block: ShiftBasicBlock,
#                  num_blocks, in_channels=3, num_classes=10, features=False):
#         super(ShiftResNet, self).__init__()
#         self.in_planes = 64
#         self.features = features

#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block: ShiftBasicBlock,
#                     planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)

#         if self.features and self.training:
#             feats = []
#             out = self.layer2(out)
#             feats.append(out)
#             out = self.layer3(out)
#             feats.append(out)
#             out = self.layer4(out)
#             feats.append(out)
#         else:
#             out = self.layer2(out)
#             out = self.layer3(out)
#             out = self.layer4(out)
#         # out = F.avg_pool2d(out, 4)
#         out = F.adaptive_avg_pool2d(out, 1)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)

#         if self.features and self.training:
#             return out, feats
#         else:
#             return out


class GhostResNet(nn.Module):
    def __init__(self, block: GhostBasicBlock,
                 num_blocks, in_channels=3, num_classes=10, ratio=4):
        super(GhostResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, ratio)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, ratio)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, ratio)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, ratio)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, ratio):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, ratio=ratio))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


# Basic Residual Block based models
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def monomial_resnet18(**kwargs):
    return MonomialResNet(MonomialBasicBlock, [2, 2, 2, 2], **kwargs)


def gaussian_resnet18(**kwargs):
    return RBFFamilyResNet(RBFFamilyBasicBlock, [2, 2, 2, 2], rbf='gaussian', **kwargs)


def multiquadric_resnet18(**kwargs):
    return RBFFamilyResNet(RBFFamilyBasicBlock, [2, 2, 2, 2], rbf='multiquadric', **kwargs)


def inverse_quadratic_resnet18(**kwargs):
    return RBFFamilyResNet(RBFFamilyBasicBlock, [2, 2, 2, 2], rbf='inverse_quadratic', **kwargs)


def inverse_multiquadric_resnet18(**kwargs):
    return RBFFamilyResNet(RBFFamilyBasicBlock, [2, 2, 2, 2], rbf='inverse_multiquadric', **kwargs)


def local_binary_resnet18(**kwargs):
    return LocalBinaryResNet(LocalBinaryBasicBlock, [2, 2, 2, 2], **kwargs)


def perturbative_resnet18(**kwargs):
    return PerturbationResNet(PerturbationBasicBlock, [2, 2, 2, 2], **kwargs)


# def shift_resnet18(**kwargs):
#     return ShiftResNet(ShiftBasicBlock, [2, 2, 2, 2], **kwargs)


def ghost_resnet18(**kwargs):
    return GhostResNet(GhostBasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def monomial_resnet34(**kwargs):
    return MonomialResNet(MonomialBasicBlock, [3, 4, 6, 3], **kwargs)


def local_binary_resnet34(**kwargs):
    return LocalBinaryResNet(LocalBinaryBasicBlock, [3, 4, 6, 3], **kwargs)


def perturbative_resnet34(**kwargs):
    return PerturbationResNet(PerturbationBasicBlock, [3, 4, 6, 3], **kwargs)


# def shift_resnet34(**kwargs):
#     return ShiftResNet(ShiftBasicBlock, [3, 4, 6, 3], **kwargs)


def ghost_resnet34(**kwargs):
    return GhostResNet(GhostBasicBlock, [3, 4, 6, 3], **kwargs)


def test():
    # from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings("ignore")

    in_channels = 3  # for MNIST, 3 for SVHN and CIFAR-10
    H, W = 32, 32  # for MNIST, 32, 32 for SVHN and CIFAR-10

    model = resnet18(num_classes=10, num_terms=5, exp_range=(1, 10),
    exp_factor=[2,4,8,16], mono_bias=False, onebyone=False)

    # MonoCNNs
    # model = monomial_resnet18(
    #     in_channels=in_channels, num_terms=1, exp_range=(1, 7), exp_factor=[64, 128, 256, 512], mono_bias=False,
    #     onebyone=False)
    # model = monomial_resnet34(
    #     in_channels=in_channels, num_terms=1, exp_range=(1, 7), exp_factor=[64, 128, 256, 512], mono_bias=True,
    #     onebyone=False)

    # RBF family model
    # model = gaussian_resnet18(in_channels=in_channels, eps_range=(1, 7), exp_factor=[64, 128, 256, 512],
    #                           onebyone=False)
    # model = multiquadric_resnet18(in_channels=in_channels, eps_range=(1, 10), exp_factor=[64, 128, 256, 512],
    #                               onebyone=True)
    # model = inverse_quadratic_resnet18(in_channels=in_channels, eps_range=(1, 10), exp_factor=[64, 128, 256, 512],
    #                                    onebyone=False)
    # model = inverse_multiquadric_resnet18(in_channels=in_channels, eps_range=(1, 10), exp_factor=[64, 128, 256, 512],
    #                                       onebyone=True)

    # LBCNN
    # model = local_binary_resnet18(in_channels=in_channels, sparsity=0.1)

    # PNN
    # model = perturbative_resnet18(in_channels=in_channels, noise_level=0.1, noisy_train=False, noisy_eval=False)

    # ShiftNet
    # model = shift_resnet18(in_channels=in_channels)

    # GhostNet
    # model = ghost_resnet18(in_channels=in_channels, ratio=2, num_classes=10)

    print(model)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_count_full = sum(p.numel() for p in model.parameters())

    print(param_count / 1e6)
    print(param_count_full / 1e6)

    data = torch.rand(1, in_channels, H, W)
    y = model(data)
    try:
        for v in y[1]:
            print(v.size())
    except:
        print(y.size())

    # flops = profile_macs(model, data) / 1e6
    # print(flops)


if __name__ == '__main__':
    test()
