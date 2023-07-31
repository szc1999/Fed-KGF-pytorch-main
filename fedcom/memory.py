import torch

import horovod.torch as hvd
from horovod.torch import allreduce_

__all__ = ['Memory', 'DGCSGDMemory']


# code modified from https://github.com/sands-lab/grace/blob/master/grace_dl/torch/memory/dgc.py
class Memory:
    @staticmethod
    def initialize(*args, **kwargs):
        pass

    @staticmethod
    def compensate(tensor, *args, **kwargs):
        return tensor
    
    @staticmethod
    def update(*args, **kwargs):
        pass

    @staticmethod
    def state_dict():
        return None
    
    @staticmethod
    def load_state_dict(state_dict):
        pass


class DGCSGDMemory(Memory):
    """ Memory for momentum correction in DGC for momentum SGD optimizer"""
    def __init__(self, momentum=0.9, nesterov=False,
                 gradient_clipping=None, momentum_masking=True):
        self.gradient_clipping = gradient_clipping
        self.momentum_masking = momentum_masking

        self.momentum = momentum
        self.nesterov = nesterov
        self.momentums = {}
        self.velocities = {}
    
    def initialize(self, named_parameters):
        if hvd.rank() == 0:
            print("=> initializing dgc sgd memory")
        for name, param in named_parameters:
            self.momentums[name] = torch.zeros_like(param.data)
            self.velocities[name] = torch.zeros_like(param.data)

    def compensate(self, grad, name, accumulate=True):
        """Update the velocities with the momentums."""
        if self.gradient_clipping is not None:
            grad = self.gradient_clipping(grad)
        mmt = self.momentums[name]
        if accumulate:
            vec = self.velocities[name]
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                vec.add_(mmt).add_(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                vec.add_(mmt)
            return vec
        else:
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                return mmt.add(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                return mmt.clone()  # TODO: save this clone

    def update(self, name, ctx):
        """Update the momentums."""
        indices = ctx[0]
        if self.momentum_masking:
            self.momentums[name].view(-1).index_fill_(0, indices, 0)
        self.velocities[name].view(-1).index_fill_(0, indices, 0)

    def state_dict(self):
        return dict(momentums=self.momentums, velocities=self.velocities)

    def load_state_dict(self, state_dict):
        momentums = state_dict['momentums']
        velocities = state_dict['velocities']
        for name in self.momentums.keys():
            if name in momentums:
                self.momentums[name] = momentums[name]
                self.velocities[name] = velocities[name]



class DgcMemory(Memory):
    def __init__(self, momentum, gradient_clipping):
        self.gradient_clipping = gradient_clipping
        self.momentum = momentum
        self.gradients = {}
        self.residuals = {}

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if self.gradient_clipping:
            tensor_squ_sum = torch.sum(tensor * tensor)
            clipping_val = torch.sqrt(allreduce_(tensor_squ_sum, average=True, name=name))
            tensor = tensor.clamp(-clipping_val, clipping_val)
        if name in self.residuals:
            self.residuals[name] = self.momentum * self.residuals[name] + tensor
        else:
            self.residuals[name] = tensor
        if name in self.gradients:
            self.gradients[name] += self.residuals[name]
            tensor = self.gradients[name]
        else:
            self.gradients[name] = tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        shape, mask, _ = ctx
        not_mask = ~mask.view(shape)
        temp = self.residuals[name] * not_mask
        self.residuals[name] = temp
        temp = self.gradients[name] * not_mask
        self.gradients[name] = temp