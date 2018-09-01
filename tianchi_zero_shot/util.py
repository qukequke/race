from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
import pandas as pd
import os


def get_dict(mode):
    all_label_list = []
    name_to_label = {}
    word_vec = {}
    label_word = {}
    word_label = {}
    raw_new_label = {}
    new_raw_label = {}
    label_to_att = OrderedDict()
    dir_1 = 'data/DatasetA_train_20180813/train.txt'
    dir_2 = 'data/DatasetA_train_20180813/attributes_per_class.txt'
    dir_3 = 'data/DatasetA_train_20180813/class_wordembeddings.txt'
    dir_4 = 'data/DatasetA_train_20180813/label_list.txt'
    root = 'data/DatasetA_train_20180813/train.txt'

    with open(root) as f:
        raw_label_list = []
        for line in f:
            raw_label = line.split()[1]
            raw_label_list.append(raw_label)
    raw_label_list = list(set(raw_label_list))
    raw_label_list = (sorted(raw_label_list, key=lambda x:int(x.replace('ZJL', ''))))
    for i, x in enumerate(raw_label_list):
        raw_new_label[x] = i
        new_raw_label[i] = x

    with open(dir_4) as f:
        for line in f:
            line = line.split()
            label_word[line[0]] = line[1]
            word_label[line[1]] = line[0]

    with open(dir_3) as f:
        for line in f:
            line = line.split()
            word_vec[line[0]] = np.array(list((map(lambda x: float(x), line[1:]))))

    if mode == 'train':
        with open(dir_1) as f:
            for i, x in enumerate(f):
                x = x.split()
                name_to_label[x[0]] = x[1]
                # label_to_name[x[1]] = x[0]
                if x[1] not in all_label_list:
                    all_label_list.append(x[1])
    with open(dir_2) as f:
        for i, x in enumerate(f):
            all_ = x.split()
            word = all_[0]
            vec = all_[1:]
    #         print(vec)
            vec = np.array(list(map(lambda x: float(x), vec)))
            label_to_att[word] = vec
    return all_label_list, name_to_label, label_to_att, word_vec, label_word, word_label, raw_new_label, new_raw_label


def get_arr_mat(label_to_att):
    arr_mat = np.zeros((230, 30))
    for i, arr in enumerate(label_to_att.values()):
        arr_mat[i, :] = arr
    # print(arr_mat.shape)
    label_names = list(label_to_att.keys())
    return arr_mat, label_names


def plot_1_list(list_, title, save_dir):
    plt.figure()
    plt.plot(list_)
    plt.title(title)
    plt.savefig(save_dir)
    plt.close()


def plot_2_list(list_1, list_2, title, save_dir):
    plt.figure()
    plt.plot(list_1)
    plt.plot(list_2)
    plt.title(title)
    plt.legend(['train_acc', 'val_acc'])
    plt.savefig(save_dir)
    plt.close()

def find_queshi_value():
    queshi = sorted(list(map(lambda x:int(x[3:]), all_label_list))) #训练集中包括190类(中间有缺的)
    print('缺失值为', end='')
    tep = -1
    for i in queshi:
        if i != tep +1:
            if tep + 1 != 0:
                print(tep+1, end='')
                print(',', end='')
        tep = i

    c = sorted(list(map(lambda x:int(x[3:]), list(label_to_att.keys()))))
    tep = -1
    # print(c)
    print('缺失值为', end='')
    for i in c:
        if i != tep +1:
            if tep + 1 != 0:
                print(tep+1, end='')
                print(',', end='')
        tep = i


def mse(y_pre, tar):
    return np.sum((y_pre - tar)**2)


def get_pre_value(y_pre, arr_mat, label_names):
    y_pre = np.tile(y_pre, [230, 1])
    # print(y_pre[:5, :])
    # print(arr_mat.shape)
    # print(arr_mat[:5, :])

    loss_arr = np.sum((y_pre - arr_mat)**2, 1)
    # print('loss_arr shape ' + str(loss_arr.shape))
    # print(loss_arr)
    index = np.argmin(loss_arr)
    # print(loss_arr[index])
    # print(index)
    # print(label_names)
    # for i ,j in enumerate(label_names):
        # print(i, j)
#     print(index)

    ##change to probablity
    pro = my_softmax(-loss_arr)
    # index1 = np.argmax(pro)
    # print(index)
    # print(index1)
    return label_names[index], pro
    # return label_names[index]

def get_word_em_mat(dir_):
    # word_mat = np.genfromtxt(dir_, dtype='float32')[:, 1:]
    word_list = []
    em_list = []
    with open(dir_) as f:
        for line in f:
            line = line.split()
            word_list.append(line[0])
            em_list.append(list(map(lambda x: float(x), line[1:])))
    # print(len(em_list))
    # print(len(em_list[0]))
    return word_list, np.array(em_list)


def my_softmax(mat):
    # print(mat)
    # a = np.tile(np.sum(mat, 0), [mat.shape[0], 1])
    mat = mat.reshape(mat.shape[0], 1)
    ret = np.exp(mat) / np.sum(np.exp(mat), 0)
    ret = ret.reshape(ret.shape[0], )
    # cc = np.exp(mat) / np.sum(np.exp(mat), 0)
    # print(cc)
    return ret

def restore_model(model_name):
    print('restore  ' + str(model_name))
    model = torch.load(model_name)
    return model


def save_features(model, dataset, dataloader, save_name, batch_size):
    feature_mat = np.zeros((len(dataset), 1024))
    print(feature_mat.shape)

    # for i, (x, y) in enumerate(tqdm(all_file_dataloader)):
    for i, (x, y) in enumerate(tqdm(dataloader)):
        # print(x.shape)
        # print(y)
        x, y = Variable(x), Variable(y)
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        features = model(x)
        features = F.avg_pool2d(features, 7, 1).squeeze()
        features = features.cpu().data.numpy()
        # print(features.shape)
        try:
            feature_mat[i*batch_size:i*batch_size + batch_size, :] = features
        except Exception:
            feature_mat[i*batch_size:, :] = features
    sio.savemat(save_name,{'features':feature_mat})


def csv_txt(dir_, save_dir):
    df = pd.read_csv(dir_)
    filenames = list(df['filename'])
    index = list(df['index'])
    # print([x for x in index if int(x.replace('ZJL', '')) > 200])

    list_ = list(map(lambda x,y:x+'\t'+y + '\n', filenames, index))
    with open(save_dir, 'w') as f:
        f.writelines(list_)

if __name__ == '__main__':
    csv_txt('submit_densenet.csv', 'submit_densenet.txt')


    # a = np.array([[1,2,3], [4,1,2], [4,4,9]]) 
    # a = -a
    # ret = my_softmax(a)
    # print(ret)


