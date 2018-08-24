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
# import cv2
# len(os.listdir('data/DatasetA_train_20180813/train'))#总共的图片个数
# list(map(lambda x, y:(x**2, y**2), [1,2,3], [2,3,4]))


# mode = 'train'
# mode = 'test_yuyi'
mode = 'test_embedding'

root_test = 'data/DatasetA_test_20180813/test/*.jpg'
root_train = 'data/DatasetA_train_20180813/train/*.jpeg'

#load zidian name:文件名,  label:ZJL,  att:(30, )
all_label_list, name_to_label, label_to_att, word_vec, label_word, word_label = get_dict(mode=mode)

# ## 开始torch模型部分
transformer = transforms.Compose([transforms.Resize([224, 224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
datasets = ZhijiangDatasets(root='data/DatasetA_train_20180813/train/*.jpeg', 
                            mode=mode,
                            transform=transformer,
                            name_to_label=name_to_label,
                            label_to_att=label_to_att,
                            word_vec=word_vec,
                            label_word=label_word)
data_loader = torch.utils.data.DataLoader(datasets, batch_size=10)

test_data_loader = torch.utils.data.DataLoader(datasets, batch_size=10)
# for i in range(100):
#     x_test, y_test = next(iter(test_data_loader))
#     print(x_test.shape)
#     print(y_test.shape)
#     break
    # print(y_test.shape)
# x, y = next(iter(data_loader))

model = vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(
                            torch.nn.Linear(7*7*512, 4096),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.5),
                            torch.nn.Linear(4096, 4096),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.5),
                            torch.nn.Linear(4096, 300))
                            # torch.nn.Sigmoid())
if torch.cuda.is_available():
    model = model.cuda()


loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


arr_mat, label_names = get_arr_mat(label_to_att)
if mode == 'train':
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
        plot_1_list(loss_list, 'epoch_loss', 'epoch_loss1.png')
        print('epoch loss ' + str((loss_epoch / i)))
        print('epoch: ' + str(epoch))
        torch.save(model, 'models/my_model2.pkl')

    
elif mode == 'test_yuyi':
    print('testing')
    # model_dir = 'models/my_model.pkl'
    model_dir = 'models/my_model2.pkl'
    all_label_list, name_to_label, label_to_att, word_vec, label_word = get_dict(mode=mode)

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
    print('test_embedding')
    # model_dir = 'models/my_model.pkl'
    model_dir = 'models/my_model2.pkl'
    all_label_list, name_to_label, label_to_att, word_vec, label_word, word_label = get_dict(mode=mode)
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


    word_list, em_mat = get_word_em_mat('data/DatasetA_train_20180813/class_wordembeddings.txt')

    iterator = iter(test_data_loader)
    f = open('my_submit1111.txt', 'w+')
    all_filenames = glob.glob(root_test)
    for filename in tqdm(all_filenames):
    # for num, filename in enumerate(tqdm(all_filenames)):
        base_name = os.path.basename(filename)
        f.write(base_name)
        f.write('\t')
        # x = cv2.imread(filename)
        # x = cv2.resize(x, (224, 224))
        # x = transforms.Normalize(x)
        # x = torch.FloatTensor(x)
        # x = x.permute(2, 1, 0)
        # x = x.view(1, x.shape[0], x.shape[1], x.shape[2])


        x, y = next(iterator)
        x = x.cuda()
        y_pre = model(x)
        # print(y_pre)
        # min_label = get_pre_value(y_pre.cpu().data.numpy()[0, :], arr_mat, label_names)
        min_label = get_pre_value(y_pre.cpu().data.numpy()[0, :], em_mat, word_list)
        # print(min_label)
        # print(word_label)
        min_label = word_label[min_label]
        # print(min_label)
        f.write(min_label)
        f.write('\n')
        # if num % 20 == 0:
            # print(num)


