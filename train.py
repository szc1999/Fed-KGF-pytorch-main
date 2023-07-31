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

from datasets.cifar import CorruptedCIFAR
from datasets.cifar import AdditioanlCIFAR10
from utils import Normalize, cross_entropy_loss_with_soft_target

torch.manual_seed(1)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--data', default='../data/cifar', type=str, help='path to CIFAR-10/100 data')
# robustness evaluation related settings
parser.add_argument('--additional-data', default='../data/cifar/CIFAR-10.1', type=str, help='path to additional CIFAR-10 test data')
parser.add_argument('--corrupt-data', default='../data/cifar/CIFAR-10-C', type=str,
                    help='path to corrputed CIFAR-10-C / CIFAR-100-C test data')
parser.add_argument('--attacks', default='fgsm#pgd#square', type=str, help='name of the output dir to save')
parser.add_argument('--cifar100', default=False, help='use CIFAR-100 dataset')
parser.add_argument('--model', default='resnet18', type=str,
                    help='which model')
parser.add_argument('--variant', default=None, type=str, help='which variant')
# polynomial transformation CNN settings
parser.add_argument('--num-terms', default=8, type=int, help='number of poly terms')
parser.add_argument('--exp-range-lb', default=1, type=int, help='exponent min')
parser.add_argument('--exp-range-ub', default=10, type=int, help='exponent max')
parser.add_argument('--exp-factor', default=32, type=int, help='fan-out factor')
parser.add_argument('--mono-bias', default=False, help='add a bias term to poly trans')
parser.add_argument('--onebyone', default=False,
                    help='whether to use 1x1 conv to blend features')
parser.add_argument('--checkpoint', default=None, type=str,
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
# knowledge distillation related
parser.add_argument('--kd', default=None, type=str,#None
                    help='path to the pretrained teacher model for knowledge distillation')
parser.add_argument('--kd-ratio', default=0.1, type=float, help='kd')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_seed = 64//args.exp_factor
print(f'{args.num_terms}seed:{num_seed},model:{args.model},variant:{args.variant},cifar100:{args.cifar100},onebyone:{args.onebyone},teacher:{args.kd}')
print('==> Preparing data..')

list_acc = []
list_loss = []
mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

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
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

corruption_types = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
    'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
    'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter',
    'speckle_noise', 'zoom_blur'
]


# Model
print('==> Building model..')
if args.model == 'resnet18':
    if args.variant == 'poly':#为resnet18时，根据args.variant的不同选择相应的模型
        from models import polynomial_resnet18
        exp_factor = [args.exp_factor, 2*args.exp_factor, 4*args.exp_factor, 8*args.exp_factor]
        net = polynomial_resnet18(
            num_classes=num_classes, num_terms=args.num_terms, exp_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, poly_bias=args.poly_bias, noise_level=args.noise_level,
            noisy_train=args.noisy_train, noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'mono':
        from models import monomial_resnet18
        exp_factor = [args.exp_factor, 2*args.exp_factor, 4*args.exp_factor, 8*args.exp_factor]
        net = monomial_resnet18(
            num_classes=num_classes, num_terms=args.num_terms, exp_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, mono_bias=args.mono_bias, onebyone=args.onebyone) 

    elif args.variant == 'polyv2':
        from models import polynomial_resnet18v2
        exp_factor = [args.exp_factor, 2 * args.exp_factor, 4 * args.exp_factor, 8 * args.exp_factor]
        net = polynomial_resnet18v2(
            num_classes=num_classes, num_terms=args.num_terms, exp_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, poly_bias=args.poly_bias, noise_level=args.noise_level,
            noisy_train=args.noisy_train, noisy_eval=args.noisy_eval)

    elif args.variant == 'gaussian':
        from models import gaussian_resnet18
        exp_factor = [args.exp_factor, 2 * args.exp_factor, 4 * args.exp_factor, 8 * args.exp_factor]
        net = gaussian_resnet18(num_classes=num_classes, eps_range=(args.exp_range_lb, args.exp_range_ub),
                                exp_factor=exp_factor, noise_level=args.noise_level, noisy_train=args.noisy_train,
                                noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'multiquadric':
        from models import multiquadric_resnet18
        exp_factor = [args.exp_factor, 2 * args.exp_factor, 4 * args.exp_factor, 8 * args.exp_factor]
        net = multiquadric_resnet18(num_classes=num_classes, eps_range=(args.exp_range_lb, args.exp_range_ub),
                                    exp_factor=exp_factor, noise_level=args.noise_level, noisy_train=args.noisy_train,
                                    noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'inverse_quadratic':
        from models import inverse_quadratic_resnet18
        exp_factor = [args.exp_factor, 2 * args.exp_factor, 4 * args.exp_factor, 8 * args.exp_factor]
        net = inverse_quadratic_resnet18(
            num_classes=num_classes, eps_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, noise_level=args.noise_level, noisy_train=args.noisy_train,
            noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'inverse_multiquadric':
        from models import inverse_multiquadric_resnet18
        exp_factor = [args.exp_factor, 2 * args.exp_factor, 4 * args.exp_factor, 8 * args.exp_factor]
        net = inverse_multiquadric_resnet18(
            num_classes=num_classes, eps_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, noise_level=args.noise_level, noisy_train=args.noisy_train,
            noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'local_binary':
        from models import local_binary_resnet18
        net = local_binary_resnet18(num_classes=num_classes, sparsity=args.sparsity)

    elif args.variant == 'perturbative':
        from models import perturbative_resnet18
        net = perturbative_resnet18(num_classes=num_classes, noise_level=args.noise_level,
                                    noisy_train=args.noisy_train, noisy_eval=args.noisy_eval)

    # elif args.variant == 'shift':
    #     from models import shift_resnet18
    #     net = shift_resnet18(num_classes=num_classes)

    elif args.variant == 'ghost':
        from models import ghost_resnet18
        net = ghost_resnet18(num_classes=num_classes, ratio=4)

    else:
        from models import resnet18
        net = resnet18(num_classes=num_classes)

    if args.kd:#如果 args.kd 不为空（表示进行知识蒸馏），则加载教师模型并进行蒸馏训练
        from models import resnet18
        teacher = resnet18(num_classes=num_classes, features=True)
        ckpt = torch.load(args.kd, map_location='cpu')# checkpoint
        teacher = teacher.to(device)
        teacher.load_state_dict(ckpt)
        print("teacher model loaded, kd_ratio = {}".format(args.kd_ratio))
        net.features = True#将 net 模型的 features 属性设置为 True，表示在训练中使用特征提取部分

elif args.model == 'resnet34':
    if args.variant == 'poly':
        from models import polynomial_resnet34
        exp_factor = [args.exp_factor, 2*args.exp_factor, 4*args.exp_factor, 8*args.exp_factor]
        net = polynomial_resnet34(
            num_classes=num_classes, num_terms=args.num_terms, exp_range=(args.exp_range_lb, args.exp_range_ub),
            exp_factor=exp_factor, poly_bias=args.poly_bias, noise_level=args.noise_level,
            noisy_train=args.noisy_train, noisy_eval=args.noisy_eval, onebyone=args.onebyone)

    elif args.variant == 'local_binary':
        from models import local_binary_resnet34
        net = local_binary_resnet34(num_classes=num_classes, sparsity=args.sparsity)

    elif args.variant == 'perturbative':
        from models import perturbative_resnet34
        net = perturbative_resnet34(num_classes=num_classes, noise_level=args.noise_level,
                                    noisy_train=args.noisy_train, noisy_eval=args.noisy_eval)

    # elif args.variant == 'shift':
    #     from models import shift_resnet34
    #     net = shift_resnet34(num_classes=num_classes)

    elif args.variant == 'ghost':
        from models import ghost_resnet34
        net = ghost_resnet34(num_classes=num_classes, ratio=4)

    else:
        from models import resnet34
        net = resnet34(num_classes=num_classes)

    if args.kd:
        from models import resnet34
        teacher = resnet34(num_classes=num_classes, features=True)
        ckpt = torch.load(args.kd, map_location='cpu')
        teacher = teacher.to(device)
        teacher.load_state_dict(ckpt)
        print("teacher model loaded, kd_ratio = {}".format(args.kd_ratio))
        net.features = True

else:
    raise NotImplementedError

net = net.to(device)

# checkpoint = torch.load(args.checkpoint, map_location='cpu')
# net.load_state_dict(checkpoint)

# print(net)

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

n_epoch = args.epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)


# Training
def train(epoch):
    if epoch % 20 == 0:print('\nEpoch: {:d}  lr: {:.4f}'.format(epoch, scheduler.get_last_lr()[0]))
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)#输入数据和目标标签
        optimizer.zero_grad()
        #net.load_state_dict(torch.load(args.checkpoint))
        if args.kd:
            outputs, feats = net(inputs)#则在前向传播过程中同时获取模型输出和特征
        else:
            outputs = net(inputs)

        if args.kd:
            with torch.no_grad():#利用教师模型在输入数据上进行前向传播，得到软标签和软特征（不反向，no_grad）
                soft_logits, soft_feats = teacher(inputs)
                soft_logits = soft_logits.detach()
                soft_feats = [feat.detach() for feat in soft_feats]
                soft_label = F.softmax(soft_logits, dim=1)

            kd_loss = cross_entropy_loss_with_soft_target(outputs, soft_label)
            loss = 0.4 * kd_loss + criterion(outputs, targets)#知识蒸馏损失kd_loss和交叉熵损失criterion的加权和作为总损失loss。其中，知识蒸馏损失的权重为0.4

            # for (_f, _sf) in zip(feats, soft_feats):#计算特征间的均方误差损失，并乘以 args.kd_ratio 加到总损失中
            #     loss += args.kd_ratio * F.mse_loss(_f, _sf)

        else:
            loss = criterion(outputs, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, net.parameters()), max_norm=5)#使用 nn.utils.clip_grad_norm_ 函数对梯度进行裁剪，以防止梯度爆炸问题
        optimizer.step()

        #累计计算训练损失、正确预测的样本数以及总样本数
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if torch.isnan(torch.tensor(loss.item())):#如果损失出现 NaN（不是一个数字）的情况，函数返回 True，表示训练过程异常终止
            return True
        data_dict = net.state_dict()
        # for k, v in data_dict.items():
        #     if 'mono' in k and '_q'in k:
        #         v2 = data_dict[k[:-16]+'.mono_exponent']
        #         data_dict[k] = v2
        net.load_state_dict(data_dict)
    print('Train Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
        train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    torch.save(net.state_dict(), f'{args.model}_{args.variant}_seed{num_seed}_cifar10_onebyone.pth')
    return False


def test(net, testloader):
    global best_acc
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


save_data_dict = OrderedDict()#保存训练过程中的数据和模型状态（acc, state_dict）
for epoch in range(start_epoch, start_epoch+n_epoch):
    terminate = train(epoch)#调用 train(epoch) 函数进行模型训练，并返回terminate，表示是否终止训练
    acc, loss, sampler_indices = test(net, testloader)#使用 test函数评估模型在测试数据集上的准确率，并返回准确率 acc 和样本索引 sampler_indices
    list_acc.append(acc)
    list_loss.append(loss)
    # corrupt_accs = []
    # for c_type in corruption_types:#针对每种损坏类型（corruption type）进行循环
    #     print('corruption type: {}'.format(c_type))
    #     corrupt_testset = CorruptedCIFAR(root=args.corrupt_data, corruption_type=c_type, transform=transform_test)#创建相应的损坏测试数据集（corrupt_testset）
    #     corrupt_testloader = torch.utils.data.DataLoader(corrupt_testset, batch_size=100, shuffle=False, num_workers=2)

    #     corrupt_acc, _ = test(net, corrupt_testloader)#使用 test函数评估模型在损坏测试数据集上的准确率
    #     corrupt_accs.append(corrupt_acc)#将结果保存在列表 corrupt_accs

    # print(np.mean(corrupt_accs))#平均准确率

    scheduler.step()#更新学习率

    if terminate:#满足终止条件，跳出训练循环
        break

# import pandas as pd
# list_acc = pd.DataFrame(list_acc)
# list_loss = pd.DataFrame(list_loss)
# list_acc.to_csv(f'./result/acc/factor{args.exp_factor}_acc.csv', encoding='gbk')
# list_loss.to_csv(f'./result/loss/factor{args.exp_factor}_loss.csv', encoding='gbk')

# save_data_dict['acc'] = acc
# save_data_dict['state_dict'] = net.state_dict()

# evaluate for OOD Robustness(鲁棒性)
if not args.cifar100:#如果 args.cifar100 为 False，表示使用的是 CIFAR-10 数据集
    print("\nEvaluating on additional CIFAR-10 testset")
    additional_testset = AdditioanlCIFAR10(root=args.additional_data, transform=transform_test)
    additional_testloader = torch.utils.data.DataLoader(
        additional_testset, batch_size=100, shuffle=False, num_workers=2)
    additional_acc, additional_loss, sampler_indices = test(net, additional_testloader)
    print(f'acc:{additional_acc},  loss:{additional_loss}')
    save_data_dict['additional_acc'] = additional_acc

print("\nEvaluating on corrupted CIFAR-10 testset")
for c_type in corruption_types:
    print('corruption type: {}'.format(c_type))
    corrupt_testset = CorruptedCIFAR(root=args.corrupt_data, corruption_type=c_type, transform=transform_test)
    corrupt_testloader = torch.utils.data.DataLoader(
        corrupt_testset, batch_size=100, shuffle=False, num_workers=2)

    corrupt_acc, corrupt_loss, sampler_indices = test(net, corrupt_testloader)
    print(f'acc:{corrupt_acc},  loss:{corrupt_loss}')

    save_data_dict['{}_acc'.format(c_type)] = corrupt_acc
