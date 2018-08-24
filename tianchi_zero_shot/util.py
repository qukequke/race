from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


def get_dict(mode):
    all_label_list = []
    name_to_label = {}
    word_vec = {}
    label_word = {}
    word_label = {}
    label_to_att = OrderedDict()
    dir_1 = 'data/DatasetA_train_20180813/train.txt'
    dir_2 = 'data/DatasetA_train_20180813/attributes_per_class.txt'
    dir_3 = 'data/DatasetA_train_20180813/class_wordembeddings.txt'
    dir_4 = 'data/DatasetA_train_20180813/label_list.txt'

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
    return all_label_list, name_to_label, label_to_att, word_vec, label_word, word_label


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
    # print(loss_arr)
    index = np.argmin(loss_arr)
    # print(loss_arr[index])
    # print(index)
    # print(label_names)
    # for i ,j in enumerate(label_names):
        # print(i, j)
#     print(index)
    return label_names[index]

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



