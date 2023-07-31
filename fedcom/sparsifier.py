import math
import random

import torch
from memory import Memory
__all__ = ['DGCCompressor']


class DGCCompressor:
    def __init__(self, compress_ratio,memory=None,
                 sample_ratio=0.01, strided_sample=True,
                 compress_upper_bound=1.3, compress_lower_bound=0.8, max_adaptation_iters=10, resample=True,
                 fp16_values=False, int32_indices=False):

        self.fp16_values = fp16_values
        self.int32_indices = int32_indices

        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.memory = Memory if memory is None else memory
        self.sample_ratio = min(max(sample_ratio, 0.01), 1.0)# 采样率 in [0.01, 1.0]
        self.strided_sample = strided_sample
        self.compress_upper_bound = compress_upper_bound
        self.compress_lower_bound = compress_lower_bound
        self.max_adaptation_iters = max_adaptation_iters
        self.resample = resample

        self.attributes = {}

    def initialize(self, named_parameters):# 保留每层的采样、压缩信息
        for name, param in named_parameters.items():
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            if self.sample_ratio < 1.0:
                pct_numel = int(math.ceil(numel * self.sample_ratio))#采样得到的元素数量 10
                cpr_numel = int(math.ceil(2 / self.compress_ratio))#要求压缩后的梯度元素数量不能小于2个 20
                if numel <= cpr_numel:#tenosr的元素数量不足以支持压缩比例
                    sample_stride = 1
                    num_samples = numel
                else:
                    sample_stride = int(math.ceil(numel / max(pct_numel, cpr_numel) / 32)) * 32 + 1 # 49 不是50
                    num_samples = numel // sample_stride # 20
                    while num_samples < max(pct_numel, cpr_numel): #看得出num_samples必须大于pct_numel和cpr_numel
                        sample_stride = sample_stride - 8
                        num_samples = numel // sample_stride
            else:# 全采样 ratio==1
                sample_stride = 1
                num_samples = numel
            top_k_samples = int(math.ceil(num_samples * self.compress_ratio))
            num_selects = int(math.ceil(numel * self.compress_ratio))
            self.attributes[name] = (numel, shape, num_selects, num_samples, top_k_samples, sample_stride)
            '''
            —— tensor -> sample -> compress ——
            numel:tensor中元素的个数, 
            shape:tensor每一维度的尺寸(list形式), 
            num_selects:ceil(采样率 * numel), 
            num_samples:采样出的元素数量 = (numel // sample_stride) if tenosr 足以承担压缩比例 else (numel), 
            top_k_samples:num_samples进行压缩后的元素数量
            sample_stride:采样步长
            num_selects:不经过采样再压缩的元素数量
            '''

    def _sparsify(self, tensor, name):
        tensor = tensor.view(-1)
        numel, shape, num_selects, num_samples, top_k_samples, sample_stride = self.attributes[name]

        importance = tensor.abs()# 以绝对值作为重要性
        if numel == num_samples:#不采样
            samples = importance
        else:
            if self.strided_sample:#按步长采样
                sample_start = random.randint(0, sample_stride - 1)
                samples = importance[sample_start::sample_stride]
            else:#随机采样
                samples = importance[torch.randint(0, numel, (num_samples, ), device=tensor.device)]

        threshold = torch.min(torch.topk(samples, top_k_samples, 0, largest=True, sorted=False)[0])#topk的最小值做阈值
        mask = torch.ge(importance, threshold)#不是topk的归0
        indices = mask.nonzero().view(-1)#记录topk的坐标
        num_indices = indices.numel()# 数量应该是等于top_k_samples

        if numel > num_samples:
            for _ in range(self.max_adaptation_iters):
                if num_indices > num_selects:
                    if num_indices > num_selects * self.compress_upper_bound:
                        if self.resample:
                            indices = indices[
                                torch.topk(importance[indices], num_selects,
                                           0, largest=True, sorted=False)[1]
                            ]
                            break
                        else:
                            threshold = threshold * self.compress_upper_bound
                    else:
                        break
                elif num_indices < self.compress_lower_bound * num_selects:
                    threshold = threshold * self.compress_lower_bound
                else:
                    break
                mask = torch.ge(importance, threshold)
                indices = mask.nonzero().view(-1)
                num_indices = indices.numel()

        indices = indices[:num_selects]
        values = tensor[indices]
        return values, indices, numel, shape, num_selects

    def compress(self, tensor, name):
        if self.compress_ratio < 1.0 and name in self.attributes:
            # compress
            tensor_compensated = self.memory.compensate(
                tensor, name, accumulate=True)
            values, indices, numel, shape, num_selects = \
                self._sparsify(tensor_compensated, name)
            self.memory.update(name, (indices, ))
            indices = indices.view(-1, 1)
            values = values.view(-1, 1)

            ctx = (name, numel, shape, values.dtype, indices.dtype,
                   tensor.data.view(numel))
            if self.fp16_values and values.dtype.is_floating_point:
                values = values.type(torch.float16)
            if self.int32_indices and not indices.dtype.is_floating_point:
                indices = indices.type(torch.int32)
            return (values, indices), ctx
        else:
            ctx = (name, None, None, tensor.dtype, None, None)
            if self.fp16_values and tensor.dtype.is_floating_point:
                tensor = tensor.type(torch.float16)
            return tensor, ctx

    def decompress(self, tensor, ctx):
        name, numel, shape, vdtype, idtype, grad = ctx
        if self.compress_ratio < 1.0 and name in self.attributes:
            # decompress
            assert isinstance(tensor, (list, tuple))
            values, indices = tensor
            values = values.view(-1)
            indices = indices.view(-1)
            if self.fp16_values and vdtype.is_floating_point:
                values = values.type(vdtype)
            if self.int32_indices and not idtype.is_floating_point:
                indices = indices.type(idtype)
            grad.zero_().index_put_([indices], values, accumulate=True)
            return grad.view(shape)
        else:
            if self.fp16_values and vdtype.is_floating_point:
                tensor = tensor.type(vdtype)
            return self.memory.compensate(tensor, name, accumulate=False)
        
if __name__ == '__main__':
    c = DGCCompressor(1/10)
    name = 'name'
    tensor = torch.tensor(torch.randint(0,10,(10,100)))
    c.initialize({name:tensor})
    c.compress(tensor=tensor,name='n')