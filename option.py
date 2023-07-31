import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--method', default='COMGATE', type=str, choices=['slice', 'avg', 'DGC', 'COMGATE'], help='the aggregation method federated learning use')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=500, type=int, help='number of training epochs')
    parser.add_argument('--local-ep', default=1, type=int, help='number of training epochs')
    parser.add_argument('--data', default='../data/cifar', type=str, help='path to CIFAR-10/100 data')
    parser.add_argument('--iid', default=False, help='iid or no-iid dataset')
    parser.add_argument('--num-client', default=10, type=int, help='iid or no-iid dataset')
    # robustness evaluation related settings
    parser.add_argument('--additional-data', default='../data/cifar/CIFAR-10.1', type=str, help='path to additional CIFAR-10 test data')
    parser.add_argument('--corrupt-data', default='../data/cifar/CIFAR-10-C', type=str, help='path to corrputed CIFAR-10-C / CIFAR-100-C test data')
    parser.add_argument('--attacks', default='fgsm#pgd#square', type=str, help='name of the output dir to save')
    parser.add_argument('--cifar100',  default=True, help='use CIFAR-100 dataset')
    parser.add_argument('--model', default='resnet18', type=str,help='which model')
    parser.add_argument('--variant', default=None, type=str, help='which variant')
    # polynomial transformation CNN settings
    parser.add_argument('--num-terms', default=5, type=int, help='number of poly terms')
    parser.add_argument('--exp-range-lb', default=1, type=int, help='exponent min')
    parser.add_argument('--exp-range-ub', default=10, type=int, help='exponent max')
    parser.add_argument('--exp-factor', default=4, type=int, help='fan-out factor')
    parser.add_argument('--mono-bias', default=False, help='add a bias term to poly trans')
    parser.add_argument('--onebyone', default=False, help='whether to use 1x1 conv to blend features')
    parser.add_argument('--checkpoint', default=None, type=str,help='path to checkpoint')
    # local binary CNN settings
    parser.add_argument('--compressor', default='topk', type=str)
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


    args = parser.parse_args()
    return args