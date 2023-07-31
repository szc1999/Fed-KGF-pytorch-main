import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import numpy as np
from collections import OrderedDict

from utils import Normalize, cross_entropy_loss_with_soft_target

layer_name = ['layer0_', 
              'layer1.0', 'layer1.1', 
              'layer2.0', 'layer2.1', 
              'layer3.0', 'layer3.1', 
              'layer4.0', 'layer4.1', 
              'linear'
              ]

class Server():
    def __init__(self, net, state_dict):
        self.net = net
        self._state_dict = state_dict
        self._temp_dict = OrderedDict()
        self.sorted_layer = []

        # 计算每一层的参数数量并存储到字典中
        param_counts = {L: 0 for L in layer_name}
        for name, params in self.net.named_parameters():
            for L in layer_name:
                if params.requires_grad and L in name and 'exponent' not in name:
                    param_counts[L] += params.numel()
                    break
        # 根据参数数量对所有层进行排序
        param_counts = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)
        for layer, counts in param_counts:
            self.sorted_layer.append(layer)
        # 打印每一层的参数数量
        for name, count in param_counts:
            print(f"{name}: {count}")
        #temp_dict归零
        for name, _ in self._state_dict.items():
            self._temp_dict[name] = self._state_dict[name] - self._state_dict[name]

    @property
    def state_dict(self):
        return self._state_dict
    
    @property
    def temp_dict(self):
        return self._temp_dict
    
    @state_dict.setter
    def state_dict(self, layer_dict):
        for name, param in layer_dict.items():
            self._state_dict[name] = param

    @temp_dict.setter
    def temp_dict(self, dict):
        for name, param in dict.items():
            self._temp_dict[name] = param
    
    def recruit(self):
        # shuffle
        if len(self.sorted_layer) == 0: self.sorted_layer = copy.deepcopy(layer_name)
        randidx = random.randint(0,len(self.sorted_layer)-1) if len(self.sorted_layer) > 1 else 0
        first_layer = self.sorted_layer.pop(randidx)
        # ordered
        # first_layer = self.sorted_layer.pop(0)
        # self.sorted_layer.append(first_layer)
        print(first_layer)
        return first_layer


class client():
    def __init__(self, dataloader, optimizer):
        self.feats = []  # feats是一个列表，里面是每层layer训练时输出的特征
        #self.stale_dict = net.state_dict()
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    # def download(self, state_dict):
    #     self.state_dict = state_dict

    # def upload(self, name_layer):
    #     # 上传该层参数
    #     partal_dict = OrderedDict()
    #     for name, param in self.state_dict.items():
    #         if name_layer in name and 'exponent' not in name:
    #             partal_dict[name] = param
    #     state_dict_size = sum(tensor.numel() * tensor.element_size() for tensor in partal_dict.values())# 计算大小
    #     print(f'dict_size:{state_dict_size} bytes')

    #     return partal_dict, state_dict_size

    # def Local_update(self, device, kd, kd_ratio):
    #     self.net.load_state_dict(self.state_dict)#temp_dict = copy.deepcopy(self.state_dict)
    #     train_loss = 0
    #     correct = 0
    #     total = 0
    #     torch.manual_seed(1)
    #     for batch_idx, (inputs, targets) in enumerate(self.dataloader):
    #         inputs, targets = inputs.to(device), targets.to(device)#输入数据和目标标签
    #         self.optimizer.zero_grad()
    #         # if kd:
    #         outputs, feats = self.net(inputs)

    #         # self.net.load_state_dict(self.stale_dict)
    #         # self.net.fuse = False
    #         # with torch.no_grad(): outputs_y, feats_y = self.net(inputs)#则在前向传播过程中同时获取模型输出和特征

    #         # self.net.feats_y = feats_y
    #         # self.net.load_state_dict(temp_dict)
    #         # self.net.fuse = True
    #         # outputs, feats = self.net(inputs)#则在前向传播过程中同时获取模型输出和特征
            
    #         # else:
    #         #     outputs = self.net(inputs)

    #         if kd:
    #             with torch.no_grad():#利用教师模型在输入数据上进行前向传播，得到软标签和软特征（不反向，no_grad）
    #                 soft_logits, soft_feats = self.teacher(inputs)
    #                 soft_logits = soft_logits.detach()
    #                 soft_feats = [feat.detach() for feat in soft_feats]
    #                 soft_label = F.softmax(soft_logits, dim=1)

    #             kd_loss = cross_entropy_loss_with_soft_target(outputs, soft_label)
    #             loss = 0.4 * kd_loss + self.criterion(outputs, targets)#知识蒸馏损失kd_loss和交叉熵损失criterion的加权和作为总损失loss。其中，知识蒸馏损失的权重为0.4

    #             for (_f, _sf) in zip(feats, soft_feats):#计算特征间的均方误差损失，并乘以 args.kd_ratio 加到总损失中
    #                 loss += kd_ratio * F.mse_loss(_f, _sf)

    #         else:
    #             loss = self.criterion(outputs, targets)

    #         loss.backward()
    #         nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.net.parameters()), max_norm=5)#使用 nn.utils.clip_grad_norm_ 函数对梯度进行裁剪，以防止梯度爆炸问题
    #         self.optimizer.step()
    #         #temp_dict = self.net.state_dict()

    #         #累计计算训练损失、正确预测的样本数以及总样本数
    #         train_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()
            
    #     self.feats = feats
    #     print('Train Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
    #         train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    #     if torch.isnan(torch.tensor(loss.item())):#如果损失出现 NaN（不是一个数字）的情况，函数返回 True，表示训练过程异常终止
    #         return True
    #     self.state_dict = self.net.state_dict()
    #     # self.stale_dict = self.state_dict
    #     # self.state_dict = temp_dict
    #     self.scheduler.step()#更新学习率

    #     return False
    