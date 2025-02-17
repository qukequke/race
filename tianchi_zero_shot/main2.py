import os
from collections import OrderedDict
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import vgg16
from torch.autograd import Variable
import matplotlib.pyplot as plt
from datasets import ZhijiangDatasets2
from util import *
import glob
# import cv2
# len(os.listdir('data/DatasetA_train_20180813/train'))#总共的图片个数
# list(map(lambda x, y:(x**2, y**2), [1,2,3], [2,3,4]))


# mode = 'train'
mode ='test'

#load zidian
all_label_list, name_to_label, label_to_att, word_vec = get_dict(mode=mode)

# ## 开始torch模型部分
transformer = transforms.Compose([transforms.Resize([224, 224]),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
datasets = ZhijiangDatasets2(root='data/DatasetA_train_20180813/train/*.jpeg',
                # root='data/DatasetA_train_20180813/class_wordembeddings.txt'
                            mode=mode,
                            transform=transformer,
                            name_to_label=name_to_label,
                            label_to_att=label_to_att,
                            word_vec=word_vec)
data_loader = torch.utils.data.DataLoader(datasets, batch_size=10)
# test_data_loader = torch.utils.data.DataLoader(datasets, batch_size=100)
# x_test, y_test = next(iter(test_data_loader))
# x, y = next(iter(data_loader))

model = vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(7*7*512, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(4096, 30),
    torch.nn.Sigmoid())
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
        plot_1_list(loss_list, 'epoch_loss', 'epoch_loss.png')
        print('epoch loss ' + str((loss_epoch / i).cpu().data.numpy()))
        print('epoch: ' + str(epoch))
        torch.save(model, 'models/my_model2.pkl')


else:
    print('testing')
    all_label_list, name_to_label, label_to_att = get_dict(mode='test')
    model = torch.load('models/my_model.pkl')
    model.eval()
    root = 'data/DatasetA_test_20180813/test/*.jpg'
    root2 = 'data/DatasetA_train_20180813/train/*.jpeg'
    test_datasets = ZhijiangDatasets(root=root,
                                     mode=mode,
                                     transform=transformer,
                                     name_to_label=name_to_label,
                                     label_to_att=label_to_att)
    test_data_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1)
    # print(dir(test_data_loader))
    # print(test_data_loader.dataset())

    # print(arr_mat[0, :])
    # for i in range(230):
    #     label_tmp = label_names[i]
    #     right = np.sum(label_to_att[label_tmp] == arr_mat[i, :])
    #     if right != 30:
    #         print('wrong')
    # print(arr_mat[-5:, :])
    # print(label_names[-5:])
    f = open('my_submit.txt', 'w+')
    all_filenames = glob.glob(root)
    for num, filename in enumerate(all_filenames):
        base_name = os.path.basename(filename)
        f.write(base_name)
        f.write('\t')
        # x = cv2.imread(filename)
        # x = cv2.resize(x, (224, 224))
        # x = transforms.Normalize(x)
        # x = torch.FloatTensor(x)
        # x = x.permute(2, 1, 0)
        # x = x.view(1, x.shape[0], x.shape[1], x.shape[2])


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
    # print(base_name)
    # print(name_to_label.keys())
    # label = name_to_label[base_name]
    # print(label)
    # att = label_to_att[min_label]

    # att1 = label_to_att['ZJL1']
    # att2 = label_to_att['ZJL142']
    # print(y_pre.cpu().data.numpy()[0, :].shape)
    # print(label.shape)
    # loss1 = mse(y_pre.cpu().data.numpy()[0, :], att)
    # loss2 = mse(y_pre.cpu().data.numpy()[0, :], att1)
    # loss3 = mse(y_pre.cpu().data.numpy()[0, :], att2)
    # print(loss1, loss2, loss3)
    # break
    # print(pre.shape)

    # f.read()
    # for i, (x, y) in enumerate(test_data_loader):
    #     y = y.float()
    #     if torch.cuda.is_available():
    #         x, y = x.cuda(), y.cuda()
    #     y_pre = model(x)
    #     # print(y_pre.shape)
    #     for k in range(y_pre.shape[0]):
    #         min_label = get_pre_value(y_pre.cpu().data.numpy()[0, :], arr_mat, label_names)

    # get_pre_value()
    # test_data_loader = torch.utils.data.DataLoader(test_datasets, batch_size=100)
    # x_test, y_test = next(iter(test_data_loader))
    # x, y = next(iter(data_loader))
# print(x.shape)

# pass
#     for i in range(10):
#         y_val = model(x_test[i*10:i*10+10, :, :, :])
#         for j in range(10):
#             min_label = get_pre_value(y_val.data.numpy()[j, :], label_to_att)

#     break
