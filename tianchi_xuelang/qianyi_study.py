import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import os
from sklearn import metrics
import sys


system = sys.platform #判断系统的
if system == 'win32':
    os.chdir('input')
mode = 'train'  # train用来训练, test生成csv提交结果
# mode = 'test'


print('mode = ' + mode)

#这一块是pytorch自带的的载入文件夹图片
transformer = transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  # transforms.CenterCrop(200),
                                  # transforms.RandomVerticalFlip(),
                                  # transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# train_data = {}
train_data = {x: torchvision.datasets.ImageFolder(x, transform=transformer)
              for x in ['train', 'val']}
# os.chdir('..')
print(train_data['train'].class_to_idx)
train_loader = {}
train_loader['train'] = torch.utils.data.DataLoader(train_data['train'],
                                               batch_size=10,
                                               shuffle=True)
train_loader['val'] = torch.utils.data.DataLoader(train_data['val'],
                                               batch_size=10,
                                               shuffle=True)

print('train num is ' + str(len(train_data['train'])))
print('val num is ' + str(len(train_data['val'])))
# model = models.vgg16(pretrained=True)
# print('model download complete')
# print(model)
# print(model)
if os.listdir('models'): #恢复模型
    print('restrore the model')
    model = torch.load('my_model.pkl')
else:
    print('use vgg16 model')

    # model = torch.load('vgg16.pkl')
    # model = torch.load('vgg_11_bn.pkl')
    # models.vgg16_bn(pretrained=True, batch_norm)
    # model.features[18] = torch.nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # for i in model.parameters():
        # print(i.requires_grad)
    model.classifier = torch.nn.Sequential(
        # torch.nn.Linear(25088, 2),
        torch.nn.Linear(7*7*512, 2), #vgg提取特征不变  分类层改一下  因为一直过拟合 所以我直接去掉了好几层

        # torch.nn.Linear(25088, 4096),
        # torch.nn.ReLU()
        # torch.nn.Dropout(0.6),
        # torch.nn.Linear(256, 2),
        # torch.nn.Linear(4096, 4096),
        # torch.nn.ReLU(),
        # torch.nn.Dropout(0.5)
        # torch.nn.Linear(1024, 2)
    )
    # for idx, para in enumerate(model.parameters()):
        # print(idx)
        # print(para.shape)
        # if idx < 14:
        #     para.requires_grad = False   #把前面固定

if torch.cuda.is_available(): #cpu gpu转换
    model = model.cuda()
print(model)


# def parameters():
#     '''
#     自己创建需要优化的参数生成器
#     :return:
#     '''
#     for idx, para in enumerate(model.parameters()):
#         if idx >= 14:
#             yield para
#             # print(para.shape)

# for i, j in enumerate(model.parameters()):
#     print(i)
#     print(j.requires_grad)

loss_func = torch.nn.CrossEntropyLoss()
lr = 1e-5
# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
# optimizer = torch.optim.Adam(parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# print(model)


## 建立这些列表基本都是用来画图的
epochs = 30 
plot_loss = []
best_auc = 0
auc_list = []
auc_list2 = []
train_acc_list = []
test_acc_list = []
# plt.ion()
def valling(dir_name, model):
    """
    得到网络输出   用来metrics
    0 1标签（用来算正确率）
    概率（算auc）
    label
    """
    model.eval()
    print('valling in ' + str(dir_name))
    y_pre_all = np.array(())
    test_y_all = np.array(())
    all_pro = np.array(())
    for tep_idx, [test_x, test_y] in enumerate(train_loader[dir_name]):
        if tep_idx <= 10:
            test_x, test_y = next(iter(train_loader[dir_name]))
            if torch.cuda.is_available():
                test_x, test_y = (test_x.cuda()), (test_y.cuda())

            y_out_test = model(test_x)

    # pro = F.softmax(pre_out).cpu().data.numpy()[:, 1]

            all_pro = np.append(all_pro, F.softmax(y_out_test, 0).cpu().data.numpy()[:, 1])
            # print(y_out_test)
            y_pre_test = torch.argmax(y_out_test, 1)

            y_pre_test = y_pre_test.cpu().data.numpy()
            test_y = test_y.cpu().data.numpy()

            # print(y_pre_all.shape)
            # print(y_pre_test.shape)
            y_pre_all = np.append(y_pre_all, y_pre_test)
            test_y_all = np.append(test_y_all, test_y)
            # print(y_pre_all.shape)
    return y_pre_all, test_y_all, all_pro
        # print(all_pro)

def my_metrics(pre, label, pro):
    '''
    计算auc  acc
    '''
    # print('label shape is ' + str(label.shape))
    # print('pro shape is ' + str(pro.shape))
    auc = metrics.roc_auc_score(label, pro)
    bool_arr_test = (pre == label) 
    test_acc = np.sum(bool_arr_test) / pre.size
    return auc, test_acc


def plot_list(list1, list2, dir_, title):
    '''
    画图  train 和test的acc  auc
    '''
    abs_dir = os.path.abspath(dir_)
    if not os.path.exists(os.path.dirname(abs_dir)):
        os.mkdir(os.path.dirname(abs_dir))
        print('creat dir{}'.format(abs_dir))
    plt.figure()
    plt.plot(list1, label='train')
    plt.plot(list2, label='test')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(dir_)
    plt.close()

if mode =='train':
    best_acc = 0
    plot_epoch_loss = []
    # print(model)
    for epoch in range(epochs):
        model.train()
        print('training')
        batch = 0
        epoch_loss = 0
        correct = 0
        # print(train_loader['train'])
        for data in train_loader['train']:
        # for data in train_loader['train']:
            batch += 1
            # print(data)
            x, y = data
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            y_out = model(x)
            optimizer.zero_grad()
            loss = loss_func(y_out, y)
            epoch_loss += loss
            # print(loss.data)
            # print(loss.data[0])
            loss.backward()
            optimizer.step()

            a_loss = loss.cpu().data.numpy()
            plot_loss.append(a_loss)
            plt.cla()
            plt.plot(plot_loss)
            print(a_loss)
            plt.text(0, 0.5, 'loss = %.3f' % a_loss, {'color': 'red', 'size': 15})
            plt.savefig('loss2.png')
            plt.close()
            plt.pause(0.5)


        y_pre_all, test_y_all, all_pro = valling('val', model)
        train_y_pre_all, train_test_y_all, train_all_pro = valling('train', model)

        auc, test_acc = my_metrics(y_pre_all, test_y_all, all_pro)
        train_auc, train_test_acc = my_metrics(train_y_pre_all, train_test_y_all, train_all_pro)

        train_acc_list.append(train_test_acc)
        test_acc_list.append(test_acc)

        saved_figs_dir = 'vgg11_full_32' 
        plot_list(train_acc_list, test_acc_list, os.path.join('saved_figs', saved_figs_dir, 'acc.png'), 'acc_curve')
        # plt.figure()
        # plt.plot(train_acc_list, label='train')
        # plt.plot(test_acc_list, label='test')
        # plt.legend(loc='best')
        # plt.title('acc_curve')
        # plt.savefig('saved_figs/2/1024units.png')
        # plt.close()


        auc_list.append(auc)
        auc_list2.append(train_auc)
        # print('train auc = ' + str(auc_list2))
        plot_list(auc_list2, auc_list, os.path.join('saved_figs', saved_figs_dir, 'auc.png'), 'auc_curve')
        # plt.figure()
        # plt.plot(auc_list, label='val')
        # plt.plot(auc_list2, label='train')
        # plt.legend(loc='best')
        # plt.title('auc_curve')
        # plt.savefig('saved_figs/2/auc.png')
        print('test auc = ' + str(auc))
        # print('correct nums is ' + str(np.sum(y_pre_all == test_y_all)))
        # print('all nums is ' + str(y_pre_all.size))
        # lr = (0.95 ** epoch) * lr


        best_acc = max(best_acc, test_acc) #保存最好的结果
        best_auc = max(best_auc, auc)

        print('test_acc = ' + str(test_acc * 100)[:4] + '%')
        print('train_acc = ' + str(train_test_acc * 100)[:4] + '%')
        epoch_loss = epoch_loss.cpu().data.numpy()
        print('This ' + str(epoch) + 'th epoch', 'epoch average loss = ' + str(epoch_loss/(batch)))
        plot_epoch_loss.append(epoch_loss / (batch))
        plt.figure()
        plt.plot(plot_epoch_loss)
        plt.title('epoch_loss')
        plt.savefig(os.path.join('saved_figs', saved_figs_dir, 'epoch_loss.png'))
        # plt.savefig('saved_figs/2/epoch_loss.png')
        print('lr = {}'.format(lr))
        if best_acc <= test_acc: #存正确率最高的模型
        # if best_auc <= auc:
            print('score is better  store model')
            torch.save(model, 'models/my_model.pkl')
        else:
            print("not good don't save")
        print('-' * 40)    


else:
    #用来生成提交结果
    test_data = torchvision.datasets.ImageFolder('test', transform=transformer)
    test_data_loader = torch.utils.data.DataLoader(
                                        test_data, 
                                        batch_size=10,
                                        shuffle=False)
    ret_df = pd.DataFrame(columns=['filename', 'probability'])
    # ret_df.columns = ['filename', 'probability']
    # model = torch.load('models/my_model.pkl')
    filenames = []
    for i in test_data.imgs:
        filename = os.path.basename(i[0])
        filenames.append(filename)

    # print(filenames)
    ret_df['filename'] = filenames
    for i, [x, y] in enumerate(test_data_loader):
        if torch.cuda.is_available():
            x = x.cuda()
        x = Variable(x)
        # print(x.shape)
        pre_out = model(x)
        pro = F.softmax(pre_out).cpu().data.numpy()[:, 1]
        pro = np.clip(pro, 0.000001, 0.999999)
        print('The ' + str(i*10) + ' th ' + 'row')
        # print(ret_df)
        # print(ret_df.loc[10*i: 10*i+10, :])
        # print(filenames[10*i:10*i+10])
        # ret_df.loc[10*i: 10*i+10, 'filename'] = filenames[10*i:10*i+10]
        # print(ret_df.tail())
        try:
            ret_df.iloc[10*i: 10*i+10, 1] = pro
        except Exception:
            # ret_df.to_csv('submission.csv', index=Fasle)
            ret_df.loc[10*i:, 'probability'] = pro

        # print(pro)
        # ret_df['filename'] = 
        # print(F.softmax(pre_out))
        # print(pre_out)
    # ret_df.loc[ret_df['probability'] == 1, 'probability'] = 0.999999
    ret_df = ret_df.round(6)
    print((ret_df['probability'] <= 0).sum())
    print((ret_df['probability'] >= 1).sum())
    ret_df.to_csv('outputs/submission.csv', index=False, encoding='utf-8')





