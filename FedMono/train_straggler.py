import copy
import os
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

import sys
file_dir = os.path.join(os.path.dirname(__file__), '..')
print(file_dir)
sys.path.append(file_dir)

from datasets.cifar import CorruptedCIFAR
from datasets.cifar import AdditioanlCIFAR10
from utils import Normalize, cross_entropy_loss_with_soft_target
from member import Server, client

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--method', default='mono', type=str, choices=['mono', 'avg'], help='the aggregation method federated learning use')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--local-ep', default=3, type=int, help='number of training epochs')
parser.add_argument('--data', default='../data/cifar', type=str, help='path to CIFAR-10/100 data')
# robustness evaluation related settings
parser.add_argument('--additional-data', default='../data/cifar/CIFAR-10.1', type=str, help='path to additional CIFAR-10 test data')
parser.add_argument('--corrupt-data', default='../data/cifar/CIFAR-10-C', type=str, help='path to corrputed CIFAR-10-C / CIFAR-100-C test data')
parser.add_argument('--attacks', default='fgsm#pgd#square', type=str, help='name of the output dir to save')
parser.add_argument('--cifar100', action='store_true', default=False, help='use CIFAR-100 dataset')
parser.add_argument('--model', default='resnet18', type=str,help='which model')
parser.add_argument('--variant', default='mono', type=str, help='which variant')
# polynomial transformation CNN settings
parser.add_argument('--num-terms', default=5, type=int, help='number of poly terms')
parser.add_argument('--exp-range-lb', default=0, type=int, help='exponent min')
parser.add_argument('--exp-range-ub', default=10, type=int, help='exponent max')
parser.add_argument('--exp-factor', default=8, type=int, help='fan-out factor')
parser.add_argument('--mono-bias', default=False, help='add a bias term to poly trans')
parser.add_argument('--onebyone', default=False, help='whether to use 1x1 conv to blend features')
parser.add_argument('--checkpoint', default='./seed_8.pth', type=str,
                    help='path to checkpoint')
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
parser.add_argument('--kd-ratio', default=0.1, type=float, help='kd')
# knowledge distillation related
args = parser.parse_args()
num_seed =64 // args.exp_factor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print('==> Preparing data..')

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
    trainset = torchvision.datasets.CIFAR100(
        root=args.data, train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR100(
        root=args.data, train=False, download=True, transform=transform_test)

    num_classes = 100

else:
    trainset = torchvision.datasets.CIFAR10(
        root=args.data, train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root=args.data, train=False, download=True, transform=transform_test)

    num_classes = 10

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
corruption_types = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
    'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
    'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter',
    'speckle_noise', 'zoom_blur'
]
from member import layer_name
# Model
print('==> Building model..')
from models import monomial_resnet18
exp_factor = [args.exp_factor, 2*args.exp_factor, 4*args.exp_factor, 8*args.exp_factor]
net = monomial_resnet18(
    num_classes=num_classes, num_terms=args.num_terms, exp_range=(args.exp_range_lb, args.exp_range_ub),
    exp_factor=exp_factor, mono_bias=args.mono_bias, onebyone=False) 
if args.kd:#如果 args.kd 不为空（表示进行知识蒸馏），则加载教师模型并进行蒸馏训练
    from models import resnet18
    teacher = resnet18(num_classes=num_classes, features=True)
    teacher_ckpt = torch.load(args.kd, map_location='cpu')# checkpoint
    teacher = teacher.to(device)
    teacher.load_state_dict(teacher_ckpt['state_dict'])
    print("teacher model loaded, kd_ratio = {}".format(args.kd_ratio))
    net.features = True#将 net 模型的 features 属性设置为 True，表示在训练中使用特征提取部分

net = net.to(device)

checkpoint = torch.load(args.checkpoint)
#print(net)

if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

parameters = filter(lambda p: p.requires_grad, net.parameters())# filter的第一个参数得是个函数，过滤掉不需要梯度的参数

n_epoch = args.epochs
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
torch.manual_seed(1)
server = Server(net, checkpoint)
list_client =[]
for l in range(len(layer_name)):
    list_client.append(client(dataloader=trainloader, 
                        net=copy.deepcopy(net), 
                        criterion=nn.CrossEntropyLoss(), 
                        lr=args.lr)
                    )

list_acc = []
list_loss = []

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
    net.load_state_dict(server.state_dict)
    acc, loss, sampler_indices = test(net, testloader)#使用 test函数评估模型在测试数据集上的准确率，并返回准确率 acc 和样本索引 sampler_indices
    straggler_idx = []
    if epoch % 3 == 0:
        straggler_idx = random.sample(list_client, int(0.2 * len(list_client)))
    for c in list_client:
        name_layer_this_epoch = server.recruit()# 返回参数量最大的一层
        print(idx_c)
        idx_c += 1
        if 'layer' in name_layer_this_epoch:
            c.net.quilk(name_layer_this_epoch)# un
            server.net.quilk(name_layer_this_epoch)# un
        c.download(server.state_dict)# client接收本epoch全局参数
        if c in straggler_idx and len(straggler_idx)>0:# not
            name_conv1 = name_layer_this_epoch + '.conv1.weight'
            name_conv1_q = name_layer_this_epoch + '.conv1_q.weight'
            name_conv2 = name_layer_this_epoch + '.conv2.weight'
            name_conv2_q = name_layer_this_epoch + '.conv2_q.weight'
            if 'layer' in name_layer_this_epoch and name_conv1_q in c.state_dict:
                c.net.unquilk(name_layer_this_epoch)
                server.net.unquilk(name_layer_this_epoch)
                conv1_q = c.state_dict[name_conv1_q]#_q
                conv1 = torch.chunk(conv1, 4, dim=0)[0]#conv1.repeat(4,1,1,1) _q
                c.state_dict[name_conv1] = conv1
                conv2_q = c.state_dict[name_conv2_q]#_q
                conv2 = torch.chunk(conv1_q, 4, dim=0)[0]#conv2.repeat(4,1,1,1) _q
                c.state_dict[name_conv2] = conv2
        for b in range(args.local_ep):
            terminate = c.Local_update(device)#client进行本地更新
        if terminate:#loss过大，跳过
            continue
        
        server.receive(c.upload(name_layer_this_epoch)) # 上传一层的参数
    # if args.method == 'avg':
    #     w_avg = copy.deepcopy(list_client[0].state_dict)
    #     for k in w_avg.keys():
    #         for c in list_client:
    #             w_avg[k] += c.state_dict[k]
    #         w_avg[k] = torch.div(w_avg[k], len(list_client))
    #     server.state_dict = w_avg

    list_acc.append(acc)
    list_loss.append(loss)

import pandas as pd
list_acc = pd.DataFrame(list_acc)
list_loss = pd.DataFrame(list_loss)
list_acc.to_csv(f'./result/acc/factor{args.exp_factor}_{args.method}_acc.csv', encoding='gbk')
list_loss.to_csv(f'./result/loss/factor{args.exp_factor}_{args.method}_loss.csv', encoding='gbk')

    # corrupt_accs = []
    # for c_type in corruption_types:#针对每种损坏类型（corruption type）进行循环
    #     corrupt_testset = CorruptedCIFAR(root=args.corrupt_data, corruption_type=c_type, transform=transform_test)#创建相应的损坏测试数据集（corrupt_testset）
    #     corrupt_testloader = torch.utils.data.DataLoader(corrupt_testset, batch_size=100, shuffle=False, num_workers=2)
    #     print('corruption type: {}'.format(c_type))
    #     corrupt_acc, _ = server.test(corrupt_testloader, device)#使用 test函数评估模型在损坏测试数据集上的准确率
    #     corrupt_accs.append(corrupt_acc)#将结果保存在列表 corrupt_accs
    # print('Epoch:{},Average_Accuracy:{}'.format(epoch, np.mean(corrupt_accs)))#平均准确率


save_data_dict = server.state_dict
torch.save(save_data_dict, f'fed_{args.model}_{args.method}_{num_seed}_model.pth')
# evaluate for OOD Robustness(鲁棒性)
if not args.cifar100:#如果 args.cifar100 为 False，表示使用的是 CIFAR-10 数据集
    print("\nEvaluating on additional CIFAR-10 testset")
    additional_testset = AdditioanlCIFAR10(root=args.additional_data, transform=transform_test)
    additional_testloader = torch.utils.data.DataLoader(additional_testset, batch_size=100, shuffle=False, num_workers=2)
    additional_acc, _ = server.test(additional_testloader, device)
    print(additional_acc)

print("\nEvaluating on corrupted CIFAR-10 testset")
for c_type in corruption_types:
    print('corruption type: {}'.format(c_type))
    corrupt_testset = CorruptedCIFAR(root=args.corrupt_data, corruption_type=c_type, transform=transform_test)
    corrupt_testloader = torch.utils.data.DataLoader(
        corrupt_testset, batch_size=100, shuffle=False, num_workers=2)

    corrupt_acc, _ = server.test(corrupt_testloader, device)
    print('{}_acc'.format(corrupt_acc))


