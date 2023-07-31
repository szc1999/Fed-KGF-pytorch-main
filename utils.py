import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def reset_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True#禁用cudnn的优化，方便复现


def cross_entropy_loss_with_soft_target(pred, soft_target):# 软目标交叉熵损失
    logsoftmax = nn.LogSoftmax()#将预测值 pred 转换为对数概率
    #计算每个样本在每个类别上的交叉熵损失
    #使用软目标与对数概率相乘，并取负值，得到交叉熵，对每个样本在所有类别上的交叉熵损失求和，平均后得到损失值
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def filter_state_dict(state_dict):#, num_seed
    if 'state_dict' in state_dict.keys():#状态字典中存在键 'state_dict'，则将状态字典更新为其值，以获取实际的模型参数字典
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()#创建一个新的有序字典 new_state_dict，用于存储过滤后的模型参数
    i = 0
    for k, v in state_dict.items():
        if 'sub_block' in k:#键字符串中包含子字符串 'sub_block'，则跳过该键，不包含在过滤后的状态字典中
            continue
        #模型训练过程中使用了 nn.DataParallel，则模型保存的 state_dict 中的键会包含 'module.' 前缀
        if 'module' not in k:#包含子字符串 'module'，则将新的键设置为去除 'module' 前缀的部分，并将对应的值存储到新的状态字典中
            new_state_dict['module.'+k] = v
        else:#不包含子字符串 'module'，则直接将键值对存储到新的状态字典中
            new_state_dict[k] = v
        # if 'conv' in k and 'layer' in k and 'mono' not in k and v.shape[1] == v.shape[0] :
        #     time = int(v.shape[1]/num_seed[i])
        #     v = torch.chunk(v, time, dim=0)
        #     new_state_dict[k] = v
        # if 'conv' in k and 'layer' in k and 'mono' not in k and v.shape[1] > v.shape[0] :
        #     time = int(v.shape[1]/v.shape[0])
        #     v = v.repeat(time, 1, 1, 1)
        #     new_state_dict[k] = v
        # i += 1
    return new_state_dict

def get_grad(server_dict, client_dict):
    diff_norm = {}
    diff_percent = 0.0
    diff_mse = 0.0
    count = 0
    l1, l2 = '', ''
    for key in server_dict:
        if 'conv' in key and 'exponent' not in key:
            # MSE
            mse = F.mse_loss(client_dict[key].float(), server_dict[key].float(), reduction='sum')
            diff_mse += mse
            # L2-norm 前后权重变化量的l2范数
            diff = torch.norm(client_dict[key].float() - server_dict[key].float()) /torch.norm(server_dict[key].float())
            diff_norm[key] = diff
            print(f'{key[:7]}:\tshape{server_dict[key].shape} \tdiff_norm:{diff}')
            # percent 平均每个元素的l2变化率
            percent = torch.sum((client_dict[key].float() - server_dict[key].float()).abs()) / server_dict[key].numel()
            diff_percent += percent

            count += 1
            #print(f'{key}:\tshape{server_dict[key].shape} \tdiff_norm:{diff}, \tmse:{mse}, \tpercent_abs:{percent}')
            # if key[5] != l1 or key[7] != l2:
            #     l1 , l2 = key[5], key[7]
            #     average_mse = diff_mse/count
            #     average_diff_norm = diff_norm / count
            #     average_diff_percent = diff_percent / len(server_dict)
            #     print(f'——average:{average_mse} {average_diff_norm} {average_percentage_diff}')
            #     diff_norm = 0.0
            #     diff_percent = 0.0
            #     diff_mse = 0.0
    return diff_norm

class Normalize(torch.nn.Module):
    def __init__(self, in_channels, mean, std):
        super(Normalize, self).__init__()
        self.in_channels = in_channels
        #使用 register_buffer 方法注册了两个缓冲张量，用于存储均值和标准差
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # 以便与输入数据进行广播操作。然后，将输入数据减去均值，并除以标准差，得到归一化后的数据
        mean = self.mean.reshape(1, self.in_channels, 1, 1)
        std = self.std.reshape(1, self.in_channels, 1, 1)
        return (input - mean) / std


def update_offset_buffers(offset_buffers, local_residuals, accumulated_delta):
    for i, offset in enumerate(offset_buffers):
        offset_buffers[i] = offset + (local_residuals[i] - accumulated_delta)*(128/5000)


def add_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return z


def sub_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] - y[i])
    return z


def mult_param(alpha, x):
    z = []
    for i in range(len(x)):
        z.append(alpha*x[i])
    return z


def norm_of_param(x):
    z = 0
    for i in range(len(x)):
        z += torch.norm(x[i].flatten(0))
    return z