from collections import OrderedDict
import copy
import os
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import sys
from sampling import cifar10_noniid, cifar_iid, cifar100_noniid
file_dir = os.path.join(os.path.dirname(__file__), '..')
print(file_dir)
sys.path.append(file_dir)

from models.resnet import resnet18
from datasets.cifar import CorruptedCIFAR
from datasets.cifar import AdditioanlCIFAR10
from utils import Normalize, cross_entropy_loss_with_soft_target, get_grad
from member import Server, client, layer_name
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--method', default='slice', type=str, choices=['slice', 'avg'], help='the aggregation method federated learning use')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epochs', default=500, type=int, help='number of training epochs')
parser.add_argument('--local-ep', default=1, type=int, help='number of training epochs')
parser.add_argument('--data', default='../data/cifar', type=str, help='path to CIFAR-10/100 data')
parser.add_argument('--iid', default=False, help='iid or no-iid dataset')
parser.add_argument('--num-client', default=20, type=int, help='iid or no-iid dataset')
# robustness evaluation related settings
parser.add_argument('--additional-data', default='../data/cifar/CIFAR-10.1', type=str, help='path to additional CIFAR-10 test data')
parser.add_argument('--corrupt-data', default='../data/cifar/CIFAR-10-C', type=str, help='path to corrputed CIFAR-10-C / CIFAR-100-C test data')
parser.add_argument('--attacks', default='fgsm#pgd#square', type=str, help='name of the output dir to save')
parser.add_argument('--cifar100',  default=True, help='use CIFAR-100 dataset')
parser.add_argument('--model', default='resnet18', type=str,help='which model')
parser.add_argument('--variant', default='mono', type=str, help='which variant')
# polynomial transformation CNN settings
parser.add_argument('--num-terms', default=5, type=int, help='number of poly terms')
parser.add_argument('--exp-range-lb', default=1, type=int, help='exponent min')
parser.add_argument('--exp-range-ub', default=10, type=int, help='exponent max')
parser.add_argument('--exp-factor', default=4, type=int, help='fan-out factor')
parser.add_argument('--mono-bias', default=False, help='add a bias term to poly trans')
parser.add_argument('--onebyone', default=False, help='whether to use 1x1 conv to blend features')
parser.add_argument('--checkpoint', default=None, type=str,help='path to checkpoint')
# local binary CNN settings
parser.add_argument('--sparsity', default=0.1, type=float,
                    help='sparsity of local binary weights, higher number means more non-zero weights')
# pertubation CNN settings
parser.add_argument('--noise-level', default=0.0, type=float,
                    help='the severity of noise induced to features (PNN) and weights (PolyCNN)')
parser.add_argument('--noisy-train', action='store_true', default=False,
                    help='whether to use random noise during every training mini-batch')
parser.add_argument('--noisy-eval', action='store_true', default=False,
                    help='whether to use random noise during every evaluation mini-batch')
parser.add_argument('--output', default='tmp', type=str, help='name of the output dir to save')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--kd', default=None, type=str,
                    help='path to the pretrained teacher model for knowledge distillation')
parser.add_argument('--kd-ratio', default=0.05, type=float, help='kd')
# knowledge distillation related
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_seed = 64//args.exp_factor
print(f'iid:{args.iid}, method:{args.method}, seed:{num_seed}, model:{args.model}, variant:{args.variant}, cifar100:{args.cifar100}, onebyone:{args.onebyone},checkpoint:{args.checkpoint},teacher:{args.kd}')
print('==> Preparing data..')
mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

# data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # omit the normalization to run Adversarial attack toolbox
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # omit the normalization to run Adversarial attack toolbox
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.cifar100:
    trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
    num_classes = 100

else:
    trainset = torchvision.datasets.CIFAR10(
        root=args.data, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=args.data, train=False, download=True, transform=transform_test)
    num_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)


dataset_splits = []
if args.iid:
    dataset_list = cifar_iid(trainset,args.num_client)
else:
    if num_classes == 10: dataset_list = cifar10_noniid(trainset, args.num_client)# 根据标签将数据集划分为子集
    if num_classes == 100: dataset_list = cifar100_noniid(trainset, args.num_client)# 根据标签将数据集划分为子集
for i in range(args.num_client):# 返回列表中所有不同的值，并从大到小排序
    # 创建子集对象，并将索引传递给 Subset
    subset = torch.utils.data.Subset(trainset, dataset_list[i])
    print(len(subset))
    dataset_splits.append(subset)

# Model
print('==> Building model..')
from models import monomial_resnet18
exp_factor = [args.exp_factor, 2*args.exp_factor, 4*args.exp_factor, 8*args.exp_factor]
if args.variant == 'mono':
    net = monomial_resnet18(
        num_classes=num_classes, num_terms=args.num_terms, exp_range=(args.exp_range_lb, args.exp_range_ub),
        exp_factor=exp_factor, mono_bias=args.mono_bias, onebyone=args.onebyone, features=True) 
    #net.feats_y = [torch.zeros([128,64,32,32]).to(device),torch.zeros([128,128,16,16]).to(device),torch.zeros([128,256,8,8]).to(device),torch.zeros([128,512,4,4]).to(device)]
else:
    net = resnet18(num_classes=num_classes, features=True)

net = nn.Sequential(
    # include normalization as first layer
    Normalize(in_channels=3, mean=mean, std=std),
    net
).to(device)

if args.kd:#如果 args.kd 不为空（表示进行知识蒸馏），则加载教师模型并进行蒸馏训练
    from models import resnet18
    teacher = resnet18(num_classes=num_classes, features=True)
    teacher_ckpt = torch.load(args.kd, map_location='cpu')# checkpoint
    teacher = nn.Sequential(
        # include normalization as first layer
        Normalize(in_channels=3, mean=mean, std=std),
        teacher
    ).to(device)
    teacher.load_state_dict(teacher_ckpt)
    print("teacher model loaded, kd_ratio = {}".format(args.kd_ratio))
    net.features = True#将 net 模型的 features 属性设置为 True，表示在训练中使用特征提取部分
else:
    teacher = None

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
else:
    checkpoint = net.state_dict()
# print(net)

if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

n_epoch = args.epochs
criterion = nn.CrossEntropyLoss()

server = Server(net,checkpoint)
list_client =[]
for l in range(args.num_client):
    list_client.append(client(dataloader=torch.utils.data.DataLoader(dataset_splits[l], batch_size=args.batch_size, shuffle=True, num_workers=4), #trainloader
                        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
                        )
                    )

list_acc = []
list_loss = []
list_time = []
list_size = []

# Training
def Local_update(global_dict, client, local_ep):
    net.load_state_dict(global_dict)#temp_dict = copy.deepcopy(self.state_dict)

    for e in range(local_ep):
        train_loss = 0
        correct = 0
        total = 0
        torch.manual_seed(1)
        for batch_idx, (inputs, targets) in enumerate(client.dataloader):
            inputs, targets = inputs.to(device), targets.to(device)#输入数据和目标标签
            client.optimizer.zero_grad()
            outputs, feats = net(inputs)

            # self.net.load_state_dict(self.stale_dict)
            # self.net.fuse = False
            # with torch.no_grad(): outputs_y, feats_y = self.net(inputs)#则在前向传播过程中同时获取模型输出和特征

            # self.net.feats_y = feats_y
            # self.net.load_state_dict(temp_dict)
            # self.net.fuse = True
            # outputs, feats = self.net(inputs)#则在前向传播过程中同时获取模型输出和特征

            if args.kd:
                with torch.no_grad():#利用教师模型在输入数据上进行前向传播，得到软标签和软特征（不反向，no_grad）
                    soft_logits, soft_feats = teacher(inputs)
                    soft_logits = soft_logits.detach()
                    soft_feats = [feat.detach() for feat in soft_feats]
                    soft_label = F.softmax(soft_logits, dim=1)

                kd_loss = cross_entropy_loss_with_soft_target(outputs, soft_label)
                loss = 0.4 * kd_loss + criterion(outputs, targets)#知识蒸馏损失kd_loss和交叉熵损失criterion的加权和作为总损失loss。其中，知识蒸馏损失的权重为0.4

                for (_f, _sf) in zip(feats, soft_feats):#计算特征间的均方误差损失，并乘以 args.kd_ratio 加到总损失中
                    loss += args.kd_ratio * F.mse_loss(_f, _sf)

            else:
                loss = criterion(outputs, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, net.parameters()), max_norm=5)#使用 nn.utils.clip_grad_norm_ 函数对梯度进行裁剪，以防止梯度爆炸问题
            client.optimizer.step()

            #累计计算训练损失、正确预测的样本数以及总样本数
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        client.feats = feats
        client.scheduler.step()#更新学习率
        print('Train Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
            train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    state_dict = net.state_dict()
    if torch.isnan(torch.tensor(loss.item())):#如果损失出现 NaN（不是一个数字）的情况，函数返回 True，表示训练过程异常终止
        return True, state_dict
    else:
        return False, state_dict
    

def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    correct_sample_indices = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_sample_indices.extend(predicted.eq(targets).cpu().detach().numpy().tolist())
            correct += predicted.eq(targets).sum().item()

            if total >= 10000:
                break
    print('Test Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100. * correct / total, test_loss / (batch_idx + 1), np.where(correct_sample_indices)[0]


for epoch in range(start_epoch, start_epoch+n_epoch):
    idx_c = 0
    print(f'Epoch:{epoch}')
    net.train()
    #Training
    best_diff_dict = {name: 999.9 for name in layer_name}
    recruit_this_round = {name: 0 for name in layer_name}

    join_client = [list_client[i] for i in random.sample(range(args.num_client), len(layer_name))]
    list_dict = []
    for idx, c in enumerate(join_client):
    # for idx, c in enumerate(list_client):
        print(idx_c)
        idx_c += 1
        global_dict = server.state_dict# client接收本epoch全局参数
        terminate, local_dict= Local_update(global_dict, c, args.local_ep)#client进行本地更新,teacher,
        list_dict.append(local_dict)
        if terminate:#loss过大，跳过
            continue
        # 比较相对变化率，上传最大或最小变化率的网络层参数
        # diff_dict = get_grad(server.state_dict, c.state_dict)
        # for (name, diff_b),(_, diff)in zip(best_diff_dict.items(),diff_dict.items()):
        #     if diff < diff_b and diff > 0.0:
        #         best_diff_dict[name] = diff
        #         recruit_this_round[name] = idx
    # print(recruit_this_round)
    

    #Uploading
    if args.method == 'slice':
        # for name, idx_c in recruit_this_round.items():# diff
        #     partal_dict, state_dict_size = join_client[idx_c].upload(name)

        for d in list_dict:
            name_layer_this_epoch = server.recruit()# 返回参数量最大的一层
            partal_dict = OrderedDict()
            for name, param in d.items():
                if name_layer_this_epoch in name and 'exponent' not in name:
                    partal_dict[name] = param
            state_dict_size = sum(tensor.numel() * tensor.element_size() for tensor in partal_dict.values())

            start_time = time.perf_counter()
            server.state_dict = partal_dict # 上传一层的参数
            execution_time = time.perf_counter() - start_time
            list_time.append(execution_time)
            list_size.append(state_dict_size)

        # 非种子层进行联邦平均
        # for name in server.state_dict.keys():
        #     if 'conv' in name and 'exponent' not in name and server.state_dict[name].shape[0] > num_seed:
        #         server.state_dict[name] =join_client[0].state_dict[name]
        #         for c in join_client[1:]:
        #             server.state_dict[name] += c.state_dict[name]
        #         server.state_dict[name] = torch.div(server.state_dict[name], len(join_client))

    if args.method == 'avg':
        w_avg = copy.deepcopy(list_dict[0])
        for d in list_dict[1:]:
            state_dict_size = sum(tensor.numel() * tensor.element_size() for tensor in w_avg.values())
            #print(f'dict_size:{state_dict_size} bytes')
            start_time = time.perf_counter()
            server.temp_dict = d
            execution_time = time.perf_counter() - start_time
            print(execution_time)
            for k,v in w_avg.items():
                w_avg[k] += d[k]
            list_time.append(execution_time)
            list_size.append(state_dict_size)
        for k,v in w_avg.items():
            w_avg[k] = torch.div(w_avg[k], len(join_client))
        server.state_dict = w_avg
    net.load_state_dict(server.state_dict)
    acc, loss, sampler_indices = test(net, testloader)#使用 test函数评估模型在测试数据集上的准确率，并返回准确率 acc 和样本索引 sampler_indices


    # for i in range(1,len(join_client)-1):
    #     mse = [0.0, 0.0, 0.0, 0.0]
    #     percent = [0.0, 0.0, 0.0, 0.0]
    #     norm = [0.0, 0.0, 0.0, 0.0]
    #     idx = 0
    #     for f0, fi in zip(join_client[0].feats, join_client[i].feats): 
    #         mse[idx] += F.mse_loss(f0,fi)
    #         diff = f0 - fi
    #         norm[idx] += torch.norm(diff)
    #         percent[idx] += (diff / fi) * 100
    #         idx+=1
    #     print(f'mseloss[{i}]:{mse[0]}, {mse[1]}, {mse[2]}, {mse[3]}')
    list_acc.append(acc)
    list_loss.append(loss)
    torch.cuda.empty_cache()

import pandas as pd
list_acc = pd.DataFrame(list_acc)
list_loss = pd.DataFrame(list_loss)
# list_time = pd.DataFrame(list_time)
# list_size = pd.DataFrame(list_size)

if args.iid:
    iid = 'iid'    
else:
    iid = 'noiid'
if args.variant == 'mono':
    list_acc.to_csv(f'./result/acc/fed_seed{num_seed}_{args.method}_{args.variant}_cifar{num_classes}_{iid}_acc.csv', encoding='gbk')
    list_loss.to_csv(f'./result/loss/fed_seed{num_seed}_{args.method}_{args.variant}_cifar{num_classes}_{iid}_loss.csv', encoding='gbk')
    # list_time.to_csv(f'./result/time/fed_seed{num_seed}_{args.method}_{args.variant}_cifar100_time.csv', encoding='gbk')
    # list_time.to_csv(f'./result/size/fed_seed{num_seed}_{args.method}_{args.variant}_cifar100_size.csv', encoding='gbk')
else:
    list_acc.to_csv(f'./result/acc/fed_{args.method}_{args.variant}_cifar{num_classes}_{iid}_acc.csv', encoding='gbk')
    list_loss.to_csv(f'./result/loss/fed_{args.method}_{args.variant}_cifar{num_classes}_{iid}_loss.csv', encoding='gbk')
    # list_time.to_csv(f'./result/time/fed_seed{num_seed}_{args.method}_{args.variant}_cifar10_time.csv', encoding='gbk')
    # list_time.to_csv(f'./result/size/fed_seed{num_seed}_{args.method}_{args.variant}_cifar10_size.csv', encoding='gbk')

    # corrupt_accs = []
    # for c_type in corruption_types:#针对每种损坏类型（corruption type）进行循环
    #     corrupt_testset = CorruptedCIFAR(root=args.corrupt_data, corruption_type=c_type, transform=transform_test)#创建相应的损坏测试数据集（corrupt_testset）
    #     corrupt_testloader = torch.utils.data.DataLoader(corrupt_testset, batch_size=100, shuffle=False, num_workers=2)
    #     print('corruption type: {}'.format(c_type))
    #     corrupt_acc, _ = server.test(corrupt_testloader, device)#使用 test函数评估模型在损坏测试数据集上的准确率
    #     corrupt_accs.append(corrupt_acc)#将结果保存在列表 corrupt_accs
    # print('Epoch:{},Average_Accuracy:{}'.format(epoch, np.mean(corrupt_accs)))#平均准确率


save_data_dict = server.state_dict
if args.variant == 'mono':
    if args.cifar100:
        torch.save(save_data_dict, f'./result/fed_{args.model}_{args.method}_{args.variant}_seed{num_seed}_iid{args.iid}_cifar100.pth')
    else:
        torch.save(save_data_dict, f'./result/fed_{args.model}_{args.method}_{args.variant}_seed{num_seed}_iid{args.iid}_cifar10.pth')
else:
    if args.cifar100:
        torch.save(save_data_dict, f'./result/fed_{args.model}_{args.method}_iid{args.iid}_cifar100.pth')
    else:
        torch.save(save_data_dict, f'./result/fed_{args.model}_{args.method}_iid{args.iid}_cifar10.pth')

# evaluate for OOD Robustness(鲁棒性)
if not args.cifar100:#如果 args.cifar100 为 False，表示使用的是 CIFAR-10 数据集,要用额外的测试集测试
    print("\nEvaluating on additional CIFAR-10 testset")
    additional_testset = AdditioanlCIFAR10(root=args.additional_data, transform=transform_test)
    additional_testloader = torch.utils.data.DataLoader(additional_testset, batch_size=100, shuffle=False, num_workers=2)
    additional_acc, _ = test(test, additional_testloader)
    print(additional_acc)

print("\nEvaluating on corrupted CIFAR-10 testset")
corruption_types = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
    'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
    'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter',
    'speckle_noise', 'zoom_blur'
]
for c_type in corruption_types:
    print('corruption type: {}'.format(c_type))
    corrupt_testset = CorruptedCIFAR(root=args.corrupt_data, corruption_type=c_type, transform=transform_test)
    corrupt_testloader = torch.utils.data.DataLoader(
        corrupt_testset, batch_size=100, shuffle=False, num_workers=2)

    corrupt_acc, _ = test(net, corrupt_testloader)
    print('{}_acc'.format(corrupt_acc))


