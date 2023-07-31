import copy
import math
import random
import numpy as np

# PyTorch Libraries
import torch

from fedcom import Quantizer

SCALE_QUANTIZE, ZERO_POINT_QUANTIZE, DTYPE_QUANTIZE = 0.001, 0, torch.qint8

class UniformQuantizer(Quantizer):
    def __init__(self, quantization_level,debug_mode):
        self.quantbound = quantization_level - 1
        self.debug_mode = debug_mode

    def quantize(self, arr, w_name):
        """
        quantize a given arr array with unifrom quantization.
        """
        max_val = torch.max(arr.abs())
        sign_arr = arr.sign()
        quantized_arr = (arr/max_val)*self.quantbound
        quantized_arr = torch.abs(quantized_arr)
        quantized_arr = torch.round(quantized_arr).to(torch.int)
        
        quantized_set = dict(max_val=max_val, signs=sign_arr, quantized_arr=quantized_arr)
        
        if self.debug_mode:
            quantized_set["original_arr"] = arr.clone()
        
        return quantized_set
    
    def dequantize(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        coefficients = quantized_set["max_val"]/self.quantbound  * quantized_set["signs"] 
        dequant_arr =  coefficients * quantized_set["quantized_arr"]

        return dequant_arr

class QsgdQuantizer(Quantizer):

    def __init__(self, quantization_level,debug_mode):
        self.quantlevel = quantization_level
        self.quantbound = quantization_level - 1
        self.debug_mode = debug_mode

    def quantize(self, arr, w_name):
        norm = arr.norm()
        abs_arr = arr.abs()

        level_float = abs_arr / norm * self.quantbound 
        lower_level = level_float.floor()
        rand_variable = torch.empty_like(arr).uniform_() 
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = (lower_level + is_upper_level)
        quantized_arr = torch.round(new_level).to(torch.int8)

        sign = arr.sign()
        quantized_set = dict(norm=norm, signs=sign, quantized_arr=quantized_arr)

        if self.debug_mode:
            quantized_set["original_arr"] = arr.clone()

        return quantized_set

    def dequantize(self, quantized_set):
        coefficients = quantized_set["norm"]/self.quantbound * quantized_set["signs"]
        dequant_arr = coefficients * quantized_set["quantized_arr"]

        return dequant_arr


# A plain quantizer that does nothing. (for vanilla FedAvg)
class PlainQuantizer(Quantizer):
    def __init__(self, debug_mode):
        self.debug_mode = debug_mode

    def quantize(self, arr, w_name):
        """
        simply return the arr.
        """
        quantized_set = dict(quantized_arr=arr)
        if self.debug_mode:
            quantized_set["original_arr"] = quantized_set["quantized_arr"]
        return quantized_set
    
    def dequantize(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        dequant_arr = quantized_set["quantized_arr"]

        return dequant_arr
    
class ComQuantizer(Quantizer):
    def __init__(self, quantization_level,debug_mode):
        self.quantlevel = quantization_level
        self.quantbound = quantization_level - 1
        self.debug_mode = debug_mode

    def quantize(self, arr, w_name, num_bits=16, adaptive=True, info=None):
        qmin = -2.**(num_bits-1)
        qmax =  2.**(num_bits-1) - 1.
        if adaptive:
            min_val, max_val, mean_val = arr.min(), arr.max(), arr.mean()

            scale = (max_val - min_val) / (qmax - qmin)
            if scale == 0.0:
                scale=0.001

            initial_zero_point = qmin - (min_val - mean_val) / scale

            zero_point = 0
            if initial_zero_point < qmin:
                zero_point = qmin
            elif initial_zero_point > qmax:
                zero_point = qmax
            else:
                zero_point = initial_zero_point
            zero_point = int(zero_point)
        else:
            if info is not None:
                scale=info[0]
                zero_point=info[1]
                mean_val=info[2]
            else:
                scale=SCALE_QUANTIZE
                zero_point=ZERO_POINT_QUANTIZE
                mean_val=0.0
        
        quantized_arr = zero_point + (arr - mean_val) / scale
        quantized_arr.clamp_(qmin, qmax).round_()
        if num_bits == 8:
            quantized_arr = quantized_arr.round().char()
        elif num_bits == 16:
            quantized_arr = quantized_arr.round().to(torch.float16)
        
        quantized_set = dict(scale=scale, zero_point=zero_point, mean_val=mean_val, quantized_arr=quantized_arr)

        if self.debug_mode:
            quantized_set["original_arr"] = arr.clone()

        return quantized_set


    def dequantize(self, quantized_set):

        return quantized_set['scale'] * (quantized_set['quantized_arr'].float() - quantized_set['zero_point']) + quantized_set['mean_val']


class TopkSparsifier:
    def __init__(self, quantization_level,debug_mode,compress_ratio=1/32):
        self.compress_ratio = compress_ratio

    def quantize(self, arr, w_name, smartIdx=True):
        k = int(arr.numel() * self.compress_ratio) if 2 / self.compress_ratio < arr.numel() else arr.numel()
        quantized_set = dict()
        quantized_set["original_arr"] = copy.deepcopy(arr)
        shape = list(arr.size())
        quantized_set["shape"] = shape
        if 'conv' in w_name or 'linear' in w_name:
            # tensor = arr.view(-1)
            importance = arr.abs()# 以绝对值作为重要性
            threshold = torch.min(torch.topk(importance.view(-1), k, 0, largest=True, sorted=False)[0])#topk的最小值做阈值
            # mask = torch.ge(importance, threshold)#不是topk的归0
            # indices = mask.nonzero().view(-1)#记录topk的坐标
            # num_indices = indices.numel()# 数量应该是等于top_k_samples
            # values = tensor[indices]
            if shape[-1] == 1: #哥伦布编码表示在哪个卷积核
                golomb = 0b0
            elif shape[-1] == 3:
                golomb = 0b000000000
            mu = threshold if smartIdx else 1
            # if 'conv' in w_name and 'bias' not in w_name:
            #     for o in range(shape[0]):
            #         for i in range(shape[1]):
            #             kernel = arr[o, i]
            #             mu = torch.mean(kernel.abs())
            #             for h in range(shape[2]):
            #                 for w in range(shape[3]):
            #                     if kernel[h, w] > 0:
            #                         kernel[h, w] = mu
            #                     elif kernel[h, w] < 0:
            #                         kernel[h, w] = -mu
            #                     else:
            #                         kernel[h, w] = 0
            #                     element = arr[o, i, h, w]
            #     s = shape[-1] if 'conv' in w_name else 1
            #     n = arr.numel() if 'conv' in w_name else 0
            #     Golomb = [0 for i in range(s*s)]
            #     mu = torch.mean(arr.abs())
            #     brr = arr.view(-1)
            #     ip, kp, lp = [], [], []
            #     # for index, value in brr:
            #     #     if s == 1:
            #     #         ip.append()
            #     #     elif s > 1 and n < pow(2, s * s):
            #     #         kp.append()
            #     #     elif s > 1 and n > pow(2, s * s):
            #     #         lp.append()
            _, indices = torch.topk(arr.abs().view(-1), k)
            mask = torch.zeros_like(arr)
            mask.view(-1).index_fill_(0, indices, mu)
            if smartIdx:
                mask.mul_(torch.sign(arr))
            else:
                mask.mul_(arr)
            quantized_set['indices'] = indices
            quantized_set["quantized_arr"] = mask
        else:
            quantized_set["quantized_arr"] = quantized_set["original_arr"]
        return quantized_set

    def dequantize(self, quantized_set):
        dequant_arr = quantized_set["quantized_arr"]
        return dequant_arr

quantizer_registry = {
    "plain":        PlainQuantizer,
    "qsgd":         QsgdQuantizer,
    "uniform":      UniformQuantizer,
    "com":          ComQuantizer,
    "topk":         TopkSparsifier
}

