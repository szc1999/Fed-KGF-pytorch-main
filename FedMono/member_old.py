import copy
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from fedcom.quantizer import *
from fedcom.buffer import WeightBuffer
import torchvision
import torchvision.transforms as transforms
import LayerName
import argparse
import numpy as np
from collections import OrderedDict

from utils import Normalize, cross_entropy_loss_with_soft_target


layer_name = ['layer0', 
              'layer1.0', 'layer1.1', 
              'layer2.0', 'layer2.1', 
              'layer3.0', 'layer3.1', 
              'layer4.0', 'layer4.1', 
              'linear'
              ]

class Server():
    def __init__(self, net, state_dict, compressor):
        self.net = net
        self.state_dict = state_dict
        self.temp_dict = OrderedDict()
        self.sorted_layer = []
        self.gamma = 1.
        self.quantizer = quantizer_registry[compressor](quantization_level=2, debug_mode=True)
        self.accumulated_delta = None

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
            print(layer, counts)
        # 打印每一层的参数数量
        state_dict_size = sum(tensor.numel() * tensor.element_size() for k,tensor in self.state_dict.items() if 'mono' not in k)
        print(state_dict_size)
        #temp_dict归零
        for name, param in self.state_dict.items():
            self.temp_dict[name] = self.state_dict[name] - self.state_dict[name]

    def recruit(self):
        #random
        if len(self.sorted_layer) == 0: self.sorted_layer = copy.deepcopy(layer_name)
        randidx = random.randint(0,len(self.sorted_layer)-1) if len(self.sorted_layer) > 1 else 0
        first_layer = self.sorted_layer.pop(randidx)
        #ordered
        # first_layer = self.sorted_layer.pop(0)
        # self.sorted_layer.append(first_layer)
        print(first_layer)
        return first_layer

    def receive_slice(self, layer_dict):
        for name, param in layer_dict.items():
            self.state_dict[name] = param

    def receive_avg(self,dict):
        for name, param in dict.items():
            self.temp_dict[name] = param

    def global_step(self, local_packages, local_residual_buffers, num_client):
        """Perform a global update with collocted coded info from local users.
        """
        accumulated_delta = WeightBuffer(self.net.state_dict(), mode="zeros")#zero
        accumulated_delta_state_dict = accumulated_delta.state_dict()

        global_model_state_dict = self.net.state_dict()
        for i, package in enumerate(local_packages):
            local_residuals_state_dict = local_residual_buffers[i].state_dict()
            for j, w_name in enumerate(global_model_state_dict):
                # dequantize
                quantized_sets = package[w_name]
                dequantized_residual = self.quantizer.dequantize(quantized_sets)
                local_residuals_state_dict[w_name] = dequantized_residual
                accumulated_delta_state_dict[w_name] += dequantized_residual
            local_residual_buffers[i] = WeightBuffer(local_residuals_state_dict)
        accumulated_delta = WeightBuffer(accumulated_delta_state_dict)*(1/num_client)
        
        global_model = WeightBuffer(global_model_state_dict)
        global_model += accumulated_delta*(self.gamma)
        self.net.load_state_dict(global_model.state_dict())

        self.accumulated_delta = accumulated_delta

class client():
    def __init__(self, dataloader, net, criterion, lr, compressor):
        self.net = net
        self.feats = []  # feats是一个列表，里面是每层layer训练时输出的特征
        self.state_dict = net.state_dict()
        self.init_dict = copy.deepcopy(net.state_dict())
        self.quantizer = quantizer_registry[compressor](quantization_level=2, debug_mode=True)
        # self.tau = 80
        self.offset_on = False
        self.criterion = criterion
        self.dataloader = dataloader
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def download(self, state_dict):
        self.state_dict = state_dict
        self.init_dict = state_dict


    def upload(self, name_layer):
        # 上传该层参数
        partal_dict = OrderedDict()
        for name, param in self.state_dict.items():
            if name_layer in name and 'exponent' not in name:
                partal_dict[name] = param
        state_dict_size = sum(tensor.numel() * tensor.element_size() for tensor in partal_dict.values())# 计算大小
        print(f'dict_size:{state_dict_size} bytes')

        return partal_dict, state_dict_size

    def Local_update(self, device, offset=WeightBuffer(None)):
        self.net.load_state_dict(self.state_dict)#temp_dict = copy.deepcopy(self.state_dict)
        train_loss = 0
        correct = 0
        total = 0
        torch.manual_seed(1)
        if self.offset_on:
            offset_times_lr = offset * self.optimizer.param_groups[0]['lr']
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs, targets = inputs.to(device), targets.to(device)#输入数据和目标标签
            self.optimizer.zero_grad()
            outputs, feats = self.net(inputs)
            #outputs = self.net(inputs)

            # self.net.load_state_dict(self.stale_dict)
            # self.net.fuse = False
            # with torch.no_grad(): outputs_y, feats_y = self.net(inputs)#则在前向传播过程中同时获取模型输出和特征

            # self.net.feats_y = feats_y
            # self.net.load_state_dict(temp_dict)
            # self.net.fuse = True
            # outputs, feats = self.net(inputs)#则在前向传播过程中同时获取模型输出和特征
            
            loss = self.criterion(outputs, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.net.parameters()), max_norm=5)#使用 nn.utils.clip_grad_norm_ 函数对梯度进行裁剪，以防止梯度爆炸问题
            self.optimizer.step()
            # temp_dict = self.net.state_dict()
            if self.offset_on:  # offset_on表示是否启用偏移项来防止客户端漂移 *********
                self._offset(self.net, offset_times_lr)      # w^(c+1) = w^(c) - \eta \hat{grad} + \eta \delta

            #累计计算训练损失、正确预测的样本数以及总样本数
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        self.feats = feats
        print('Train Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
            train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if torch.isnan(torch.tensor(loss.item())):#如果损失出现 NaN（不是一个数字）的情况，函数返回 True，表示训练过程异常终止
            return True
        self.state_dict = self.net.state_dict()
        # self.stale_dict = self.state_dict
        # self.state_dict = temp_dict
        self.scheduler.step()#更新学习率

        return False
    
    def _offset(self, model, offset_times_lr):
            model_buffer = WeightBuffer(model.state_dict())
            model_buffer = model_buffer + offset_times_lr
            model.load_state_dict(model_buffer.state_dict())

    def uplink_transmit(self):
        """
            用于模拟本地更新后的模型权重与接收到的初始模型权重之间的残差传输
        """ 
        try:#检查本地模型权重是否存在
            assert(self.state_dict != None)
        except AssertionError:
            logging.error("No local model buffered!")

        # 计算本地更新后的权重与初始权重之间的差异，然后对差异进行量化和压缩
        quantized_sets = OrderedDict()

        for w_name, w_pred in self.state_dict.items():
            # if 'track' not in w_name:#'weight' in w_name or 'bias' in w_name:
            residual = self.state_dict[w_name] - self.init_dict[w_name]
            quantized_set = self.quantizer.quantize(residual, w_name)
            quantized_sets[w_name]=quantized_set

        local_package = quantized_sets
        return local_package