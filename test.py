
import math
import random
import torch
from utils import filter_state_dict
from models.resnet import monomial_resnet18, resnet18


def test():
    # from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings("ignore")

    in_channels = 3  # for MNIST, 3 for SVHN and CIFAR-10
    H, W = 32, 32  # for MNIST, 32, 32 for SVHN and CIFAR-10

    model = resnet18(in_channels=in_channels, num_classes=10)

    # MonoCNNs
    model = monomial_resnet18(
        num_classes=10, num_terms=5, exp_range=(0, 10),
        exp_factor=[32, 64, 128, 256], mono_bias=False, onebyone=False) .cuda()
    # model = monomial_resnet34(
    #     in_channels=in_channels, num_terms=1, exp_range=(1, 7), exp_factor=[64, 128, 256, 512], mono_bias=True,
    #     onebyone=False)

    # RBF family model
    # model = gaussian_resnet18(in_channels=in_channels, eps_range=(1, 7), exp_factor=[64, 128, 256, 512],
    #                           onebyone=False)
    # model = multiquadric_resnet18(in_channels=in_channels, eps_range=(1, 10), exp_factor=[64, 128, 256, 512],
    #                               onebyone=True)
    # model = inverse_quadratic_resnet18(in_channels=in_channels, eps_range=(1, 10), exp_factor=[64, 128, 256, 512],
    #                                    onebyone=False)
    # model = inverse_multiquadric_resnet18(in_channels=in_channels, eps_range=(1, 10), exp_factor=[64, 128, 256, 512],
    #                                       onebyone=True)

    # LBCNN
    # model = local_binary_resnet18(in_channels=in_channels, sparsity=0.1)

    # PNN
    # model = perturbative_resnet18(in_channels=in_channels, noise_level=0.1, noisy_train=False, noisy_eval=False)

    # ShiftNet
    # model = shift_resnet18(in_channels=in_channels)

    # GhostNet
    # model = ghost_resnet18(in_channels=in_channels, ratio=2, num_classes=10)

    print(model)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)# state_dict可学习参数的数量
    param_count_full = sum(p.numel() for p in model.parameters())# state_dict不可学习参数的数量

    print(param_count / 1e6)
    print(param_count_full / 1e6)

    data = torch.rand(1, in_channels, H, W).cuda()
    y = model(data)
    try:
        for v in y[1]:
            print(v.size())
    except:
        print(y.size())

    # flops = profile_macs(model, data) / 1e6
    # print(flops)

if __name__ == '__main__':


    dict = torch.load('result/fed_resnet18_slice_mono_seed16_iidTrue_cifar10.pth')
    dict = filter_state_dict(dict)
    for name, param in dict.items():
        # if 'conv' in name and 'weight' in name and 'gn' not in name:
        print(f'\'{name}\',{param.size()}')
    
    
    # test()
