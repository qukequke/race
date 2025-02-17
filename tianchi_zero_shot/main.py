import os
from collections import OrderedDict
import torch
# import torchvision
import numpy as np
# import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import vgg16
from torch.autograd import Variable
from datasets import ZhijiangDatasets
from util import *
import glob
from tqdm import tqdm
import pandas as pd
# import cv2
# len(os.listdir('data/DatasetA_train_20180813/train'))#总共的图片个数
# list(map(lambda x, y:(x**2, y**2), [1,2,3], [2,3,4]))


# mode = 'train'
# mode = 'test_yuyi'
mode = 'test_embedding'

root_test = 'data/DatasetA_test_20180813/test/*.jpg'
root_train = 'data/DatasetA_train_20180813/train/*.jpeg'

batch_size = 10

#load zidian name:文件名,  label:ZJL,  att:(30, )
all_label_list, name_to_label, label_to_att, word_vec, label_word, word_label, raw_new_label, new_raw_label = get_dict(mode=mode)
transformer = transforms.Compose([transforms.Resize([224, 224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


if mode == 'train':
    # ## 开始torch模型部分
    datasets = ZhijiangDatasets(root='data/DatasetA_train_20180813/train/*.jpeg', 
                                mode=mode,
                                transform=transformer,
                                name_to_label=name_to_label,
                                label_to_att=label_to_att,
                                word_vec=word_vec,
                                label_word=label_word)
    data_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size)
    test_data_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size)

    # if os.listdir('models'):
    #     model = torch.load('models/.pkl')
    #     print('continue training')
    # else:
    #     model = vgg16(pretrained=True)
    #     model.classifier = torch.nn.Sequential(
    #                                 torch.nn.Linear(7*7*512, 4096),
    #                                 torch.nn.ReLU(),
    #                                 torch.nn.Dropout(0.5),
    #                                 torch.nn.Linear(4096, 4096),
    #                                 torch.nn.ReLU(),
    #                                 torch.nn.Dropout(0.5),
    #                                 torch.nn.Linear(4096, 300))
    #                                 # torch.nn.Sigmoid())
    model = restore_model('models/densenet.pkl')
    model.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 300))

    if torch.cuda.is_available():
        model = model.cuda()


    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 40
    loss_list = []
    print('training')
    for epoch in range(epochs):
        loss_epoch = 0
        for i, (x, y) in enumerate(data_loader):
            # print(i)
            y = y.float()
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            y_pre = model(x)
            # min_label = get_pre_value(y_pre.cpu().data.numpy()[0, :], arr_mat, label_names)
            loss = loss_func(y_pre, y)
            loss_epoch += loss.cpu().data.numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss_epoch/i)
        plot_1_list(loss_list, 'epoch_loss', 'epoch_loss_mse.png')
        print('epoch loss ' + str((loss_epoch / i)))
        print('epoch: ' + str(epoch))
        torch.save(model, 'models/densenet_mse.pkl')

    
elif mode == 'test_yuyi':
    arr_mat, label_names = get_arr_mat(label_to_att)
    print('testing')
    # model_dir = 'models/my_model.pkl'
    model_dir = 'models/my_model2.pkl'
    all_label_list, name_to_label, label_to_att, word_vec, label_word, word_label = get_dict(mode=mode)
    # all_label_list, name_to_label, label_to_att, word_vec, label_word = get_dict(mode=mode)

    model = torch.load(model_dir)
    model.eval()


    test_datasets = ZhijiangDatasets(root=root_test,
                                mode=mode,
                                transform=transformer,
                                name_to_label=name_to_label,
                                label_to_att=label_to_att,
                                word_vec=word_vec,
                                label_word=label_word)

    test_data_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1)
    f = open('my_submit.txt', 'w+')
    all_filenames = glob.glob(root)
    for num, filename in enumerate(all_filenames):
        base_name = os.path.basename(filename)
        f.write(base_name)
        f.write('\t')
        x, y = next(iter(test_data_loader))
        x = x.cuda()
        y_pre = model(x)
        # print(y_pre)
        min_label = get_pre_value(y_pre.cpu().data.numpy()[0, :], arr_mat, label_names)
        f.write(min_label)
        f.write('\n')
        if num %20 == 0:
            print(num)
    print('finished')


elif mode == 'test_embedding':
    batch_size = 20
    print('test_embedding')
    arr_mat, label_names = get_arr_mat(label_to_att)
    model1_dir = 'models/my_model.pkl'
    model2_dir = 'models/my_model2.pkl'
    # model1 = torch.load(model1_dir)
    # model1.eval()
    # model2 = torch.load(model2_dir)
    model2 = restore_model('models/densenet_mse.pkl')
    model2.eval()

    test_datasets = ZhijiangDatasets(root=root_test,
                                mode=mode,
                                transform=transformer,
                                name_to_label=name_to_label,
                                label_to_att=label_to_att,
                                word_vec=word_vec,
                                label_word=label_word)

    test_data_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size)


    word_list, em_mat = get_word_em_mat('data/DatasetA_train_20180813/class_wordembeddings.txt')

    iterator = iter(test_data_loader)
    # f = open('my_submit2.txt', 'w+')
    all_filenames = glob.glob(root_test)

    # weight_list = [0.10, 0.11]
    # weight_list = weight_list / sum(weight_list)
    # print(weight_list)

    submit_df = pd.DataFrame(columns=['filename', 'index'])
    all_filenames = list(map(lambda x:os.path.basename(x), all_filenames))
    submit_df['filename'] = all_filenames
    pro_df = pd.DataFrame(columns=range(230), index=all_filenames)
    # print(submit_df.tail())
    # for i in range(len(all_filenames) // batch_size):



        # x, y = next(iterator)
        # print(x.shape)
        # print(y.shape)

    min_label_list_all = []
    pro_list = []
    for x, y in tqdm(test_data_loader):
        x = x.cuda()
        y_pre = model2(x)
        # y_pre2 = model1(x)
    #     # print(y_pre)
        # min_label = get_pre_value(y_pre.cpu().data.numpy()[j, :], arr_mat, label_names)
        min_label_list = []
        # print(y_pre.shape[0])
        for j in range(y_pre.shape[0]):
            min_label1, pro1 = get_pre_value(y_pre.cpu().data.numpy()[j, :], em_mat, word_list)
            min_label = word_label[min_label1]
            min_label_list.append(min_label)
            pro_list.append(pro1)
        min_label_list_all.extend(min_label_list)
    # print(len(min_label_list_all))

    # for k, pro_array in enumerate(pro_list):
    #     print(k)
    #     pro_df.iloc[k, :] = pro_list[k]

    # submit_df.loc[batch_size*i:batch_size*i + batch_size, 'index'] = min_label_list 
    # pro_df.to_csv('pro_densenet.csv')
    submit_df.loc[:, 'index'] = min_label_list_all
    submit_df.to_csv('submit_densenet.csv')
    # print(submit_df.tail)
    #         # print(pro1.shape)


        # min_label2, pro2 = get_pre_value(y_pre2.cpu().data.numpy()[0, :], arr_mat, label_names)

        # print(min_label)
        # print(word_label)
        # print(pro)
        # print(min_label)
        # f.write(min_label)
        # f.write('\n')
        # if num % 20 == 0:
            # print(num)


