import numpy as np

from torchvision import datasets, transforms
import torch
import random

def cifar10_noniid(dataset, num_users, main_label_prop=0.8, other=9):
    """
    non-i.i.d数据生成，设置no-iid度α，一个client中占α的数据为同一标签，1-α个数据为其余标签
    默认100个client，no-iid度α=0.8
    数量分布随机从min_train到max_train，每个client α的数据为一类图片， 1-α的为其他类图片
    """
    datasize = len(dataset) // num_users
    num_shards, num_imgs = 10, 5000  # 10类图片，每类6000张

    list_users = []
    idxs = np.arange(num_shards * num_imgs)     # 空列 索引（0~59999）
    labels = np.array(dataset.targets)  # 原始数据的标签

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # idxs:           [--------------------]    idx代表图片在原始数据集中的索引
    # idxs_labels[1]: [0, 0, 0, ... 9, 9, 9]    label代表图片对应的数字标签
    for i in range(num_users):
        # 这里的随机数量，要修改成指定的数量-->有num_users个设备(users)，对应index的那个user指定一个datasize（通过读取文件实现）
        main_label = i % 10 # 0-9按顺序选一个为主类

        # 映射表仅仅决定datasize，不决定main_label
        main_label_size = int(np.floor(datasize * main_label_prop))
        other_label_size = datasize - main_label_size

        # main label idx array
        idx_begin = np.random.randint(0, num_imgs - main_label_size) + main_label * num_imgs

        # other label idx array
        other_label_list = np.zeros(other_label_size, dtype='int64')
        other_nine_label = np.delete(np.arange(10), main_label) # 剔除主标签
        other_label_class = np.random.choice(other_nine_label, size=other, replace=False) # other指剩余标签要取几种

        count = 0
        for j in range(other_label_size):  # 怎么保证不取重复值？
            label = other_label_class[count % other]
            other_label_list[j] = idxs[int(np.random.randint(0, num_imgs) + label * num_imgs)]
            count += 1

        list_users.append(np.concatenate((idxs[idx_begin : idx_begin+main_label_size], other_label_list),axis=0))
    
    return list_users
    
def cifar100_noniid(dataset, num_users, main_label_prop=0.8, other=90):
    """
    non-i.i.d数据生成，设置no-iid度α，一个client中占α的数据为同一标签，1-α个数据为其余标签
    默认100个client，no-iid度α=0.8
    数量分布随机从min_train到max_train，每个client α的数据为一类图片， 1-α的为其他类图片
    """
    datasize = len(dataset) // num_users
    num_shards, num_imgs = 100, 500  # 10类图片，每类6000张

    list_users = []
    idxs = np.arange(num_shards * num_imgs)     # 空列 索引（0~59999）
    labels = np.array(dataset.targets)  # 原始数据的标签

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # idxs:           [--------------------]    idx代表图片在原始数据集中的索引
    # idxs_labels[1]: [0, 0, 0, ... 99, 99, 99]    label代表图片对应的数字标签
    for i in range(num_users):
        # 这里的随机数量，要修改成指定的数量-->有num_users个设备(users)，对应index的那个user指定一个datasize（通过读取文件实现）
        main_label = [i*(100 //num_users)+j for j in range(100 //num_users)] # 0-99按顺序选一个为主类

        # 映射表仅仅决定datasize，不决定main_label
        main_label_size = int(np.floor(datasize * main_label_prop))
        other_label_size = datasize - main_label_size
        main_label_size = main_label_size // (100 //num_users)
        # main label idx array
        idx_begin = [np.random.randint(0, num_imgs - main_label_size) + main_label[i] * num_imgs for i in range(100 //num_users)]

        # other label idx array
        other_label_list = np.zeros(other_label_size, dtype='int64')
        other_ninty_label = np.delete(np.arange(100), main_label) # 剔除主标签
        other_label_class = np.random.choice(other_ninty_label, size=other, replace=False) # other指剩余标签要取几种

        count = 0
        for j in range(other_label_size):  # 怎么保证不取重复值？
            label = other_label_class[count % other]
            other_label_list[j] = idxs[int(np.random.randint(0, num_imgs) + label * num_imgs)]
            count += 1

        list_idxs = [idxs[idx_begin[i] : idx_begin[i]+main_label_size] for i in range(100 //num_users)]
        list_idxs.append(other_label_list)
        list_users.append(np.concatenate(list_idxs,axis=0)) 
    return list_users

def cifar_iid(dataset, num_users):
    datasize = len(dataset) // num_users  # 指定每个user或者说设备多少块数据
    all_idxs = [i for i in range(len(dataset))]  # all_idxs是dataset分块数据后每块数据的index
    # 下面就是先 shuffle all_idxs，把dataset分块数据的index打乱
    random.shuffle(all_idxs)
    # 然后把打乱的index按顺序和num_items（每块大小）分配给各个client
    list_users = []
    for i in range(num_users):
        idx_begin = i * datasize
        list_users.append(all_idxs[(idx_begin):(idx_begin+datasize)])
    return list_users