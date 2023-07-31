from collections import OrderedDict
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import sys

file_dir = os.path.join(os.path.dirname(__file__), '..')
print(file_dir)
sys.path.append(file_dir)
from option import args_parser
from models.resnet import resnet18
from datasets.cifar import CorruptedCIFAR
from datasets.cifar import AdditioanlCIFAR10
from utils import Normalize, cross_entropy_loss_with_soft_target, get_grad, update_offset_buffers
from member_old import Server, client, layer_name
from DGC import DGC_SGD
from sampling import cifar10_noniid, cifar_iid, cifar100_noniid
from fedcom.buffer import WeightBuffer

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

args = args_parser()

# knowledge distillation related
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_seed = 64//args.exp_factor
print(f'iid:{args.iid}, method:{args.method}, seed:{num_seed}, model:{args.model}, variant:{args.variant}, cifar100:{args.cifar100}, onebyone:{args.onebyone},checkpoint:{args.checkpoint}')
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
        exp_factor=exp_factor, mono_bias=args.mono_bias, onebyone=args.onebyone,features=True) 
    #net.feats_y = [torch.zeros([128,64,32,32]).to(device),torch.zeros([128,128,16,16]).to(device),torch.zeros([128,256,8,8]).to(device),torch.zeros([128,512,4,4]).to(device)]
else:
    net = resnet18(num_classes=num_classes, features=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is available!")
    print('Memory Usage:')
    print('Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    print('cuDNN:    ', torch.backends.cudnn.version())

else:
    device = torch.device("cpu")

net = nn.Sequential(
    # include normalization as first layer
    Normalize(in_channels=3, mean=mean, std=std),
    net
).to(device)

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
else:
    checkpoint = net.state_dict()
# print(net)

if device == 'cuda':
    cudnn.benchmark = True

n_epoch = args.epochs
criterion = nn.CrossEntropyLoss()
if args.method == 'DGC':
    optimizer = DGC_SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
server = Server(net,checkpoint,args.compressor)
list_client =[]
offset_buffers = []
residual_buffers = []
for l in range(args.num_client):
    list_client.append(client(dataloader=torch.utils.data.DataLoader(dataset_splits[l], batch_size=args.batch_size, shuffle=True, num_workers=4), #trainloader
                        net=copy.deepcopy(net), 
                        criterion=nn.CrossEntropyLoss(), 
                        lr=args.lr,
                        compressor=args.compressor)
                    )
    if args.method == 'COMGATE':
        offset_buffers.append(WeightBuffer(net.state_dict(), mode="zeros"))
        residual_buffers.append(WeightBuffer(net.state_dict(), mode="zeros"))

list_acc = []
list_loss = []
list_time = []
list_size = []

def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    dict = net.state_dict()

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
    local_packages = []
    #Training
    join_client = [list_client[i] for i in random.sample(range(args.num_client), len(layer_name))]
    for idx, c in enumerate(join_client):# for idx, c in enumerate(list_client):
        print(idx_c)
        c.download(server.state_dict)# client接收本epoch全局参数
        if args.method == 'COMGATE':
            for b in range(args.local_ep):
                terminate = c.Local_update(device, offset_buffers[idx_c])
                local_package = c.uplink_transmit()
                state_dict_size = sum(tensor['quantized_arr'].numel() * tensor['quantized_arr'].element_size() for tensor in local_package.values())
                print(state_dict_size)
                local_packages.append(local_package)
        else:
            for b in range(args.local_ep):
                terminate = c.Local_update(device)#client进行本地更新
                if terminate:#loss过大，跳过
                    continue
        dd = c.state_dict
        idx_c += 1
    # print(recruit_this_round)

    #Uploading
    if args.method == 'slice':
        # for name, idx_c in recruit_this_round.items():# diff
        #     partal_dict, state_dict_size = join_client[idx_c].upload(name)

        for c in join_client:
            name_layer_this_epoch = server.recruit()# 返回参数量最大的一层
            partal_dict, state_dict_size = c.upload(name_layer_this_epoch)
            start_time = time.perf_counter()
            server.receive_slice(partal_dict) # 上传一层的参数
            execution_time = time.perf_counter() - start_time
            list_time.append(execution_time)
            list_size.append(state_dict_size)
        name_layer_this_epoch = server.recruit()

    elif args.method == 'avg' or args.method =='DGC':
        w_avg = copy.deepcopy(join_client[0].state_dict)
        start_time = time.perf_counter()
        server.receive_avg(join_client[0].state_dict)
        execution_time = time.perf_counter() - start_time
        for c in join_client[1:]:
            state_dict_size = sum(tensor.numel() * tensor.element_size() for tensor in w_avg.values())
            print(f'dict_size:{state_dict_size} bytes')
            start_time = time.perf_counter()
            server.receive_avg(c.state_dict)
            execution_time = time.perf_counter() - start_time
            print(execution_time)
            for k,v in w_avg.items():
                w_avg[k] += server.temp_dict[k]
            list_time.append(execution_time)
            list_size.append(state_dict_size)
        for k,v in w_avg.items():
            w_avg[k] = torch.div(w_avg[k], len(join_client))
        server.state_dict = w_avg


    elif args.method == 'COMGATE':
        # bn_avg = WeightBuffer(server.state_dict, mode="zeros").state_dict()
        # for c in join_client:
        #     for k,v in bn_avg.items():
        #         if 'weight' not in k and 'bias' not in k:
        #             bn_avg[k] += c.state_dict[k]
        # for k,v in bn_avg.items():
        #     if 'weight' not in k and 'bias' not in k:
        #         bn_avg[k] = torch.div(bn_avg[k], len(join_client))
        # Update the global model
        s = 0
        t_dict = OrderedDict()
        for p in local_packages:
            state_dict_size = sum(t['quantized_arr'].numel() * t['quantized_arr'].element_size() for _,t in p.items())
            s += state_dict_size
        for k,v in local_package.items():
            t_dict[k] = v['quantized_arr']
        print(s)
        start_time = time.perf_counter()
        server.receive_avg(t_dict)
        execution_time = time.perf_counter() - start_time
        print(execution_time)
        server.global_step(local_packages, residual_buffers, args.num_client)
        # Update local offsets
        update_offset_buffers(offset_buffers, residual_buffers, server.accumulated_delta) 
        # for k, v in server.state_dict.items():
        #     if 'weight' not in k and 'bias' not in k:
        #         server.state_dict[k] = bn_avg[k]

    net.load_state_dict(server.state_dict)
    acc, loss, sampler_indices = test(net, testloader)
    list_acc.append(acc)
    list_loss.append(loss)
    torch.cuda.empty_cache()
acc, loss, sampler_indices = test(net, testloader)#使用 test函数评估模型在测试数据集上的准确率，并返回准确率 acc 和样本索引 sampler_indices

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


