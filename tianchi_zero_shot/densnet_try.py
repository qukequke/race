import os
from torchvision import models
import torch
from torch.autograd import Variable
from util import *
from datasets import Densenet_Datasets
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import scipy.io as sio

mode = 'train'
epochs = 50
batch_size = 10
root_test = 'data/DatasetA_test_20180813/test/*.jpg'
root_train = 'data/DatasetA_train_20180813/train/*.jpeg'

all_label_list, name_to_label, label_to_att, word_vec, label_word, word_label, raw_new_label, new_raw_label = get_dict(mode=mode)

# with open('data/DatasetA_train_20180813/label_transform.txt', 'w') as f:
#     for i,j in raw_new_label.items():
#         f.write(str(i))
#         f.write('\t')
#         f.write(str(j))
#         f.write('\n')
transformer = transforms.Compose([# transforms.CenterCrop(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                # transforms.RandomRotation()
                                ])
dataset = Densenet_Datasets(root_train, name_to_label, 'train', raw_new_label, transformer=transformer)
dataset2 = Densenet_Datasets(root_train, name_to_label, 'val', raw_new_label, transformer=transformer)
dataset3 = Densenet_Datasets(root_train, name_to_label, 'all', raw_new_label, transformer=transformer)
dataset4 = Densenet_Datasets(root_test, name_to_label, 'test', raw_new_label, transformer=transformer)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
val_dataloader = torch.utils.data.DataLoader(dataset2, batch_size)
all_file_dataloader = torch.utils.data.DataLoader(dataset3, batch_size)
test_file_dataloader = torch.utils.data.DataLoader(dataset4, batch_size)

model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 190))

# model = models.vgg16(pretrained=True)
# model.classifier = torch.nn.Sequential(
#                             torch.nn.Linear(7*7*512, 4096),
#                             # torch.nn.Linear(2*2*512, 1024),
#                             torch.nn.ReLU(),
#                             torch.nn.Dropout(0.5),
#                             torch.nn.Linear(4096, 4096),
#                             torch.nn.ReLU(),
#                             torch.nn.Dropout(0.5),
#                             torch.nn.Linear(4096, 190))

model = restore_model('models/densenet.pkl')
model = model.features

# save_features(model, dataset3, all_file_dataloader, 'train_features.mat', batch_size)
# feature_mat = np.zeros((len(dataset3), 1024)ature_mat})
if torch.cuda.is_available():
    model = model.cuda()

# model.classifier = torch.nn.Sequential(torch.nn.Linear(4096, 190))

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy(dataloader):
    print('cacl acc')
    correction = 0
    all_num = 0
    for i, (x, y) in enumerate(tqdm(dataloader)):
    # for i, (x, y) in enumerate((dataloader)):
        # if i >= 20:
        #     break
        x, y = Variable(x), Variable(y)
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        y_pre = model(x)
        y_pre = torch.argmax(y_pre, 1)
        y_ = (y_pre == y).sum().cpu().data.numpy()
        # print('this batch right num is ' + str(y_))
        correction += y_
        all_num += y.shape[0]
    acc = correction / all_num
    return acc

epoch_loss_list = []
epoch_train_acc = []
epoch_val_acc = []
step_loss_list = []

for epoch in range(epochs):
    print('The %d epoch' % epoch)
    epoch_loss = 0
    for i, (x, y) in enumerate(train_dataloader):
        # print(x.max())
        # print(x.min())
        # print(type(x))
        # plt.imshow(x[0, 0, :, :].data.numpy())
        # plt.title(y[0].data.numpy())
        # plt.show()
        x, y = Variable(x), Variable(y)
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        y_pre = model(x)
        # print(y_pre)
        # print(y_pre.shape)
        # plt.imshow(x.cpu().data.numpy()[0, :, :, :])
        # yy = int(y[0].cpu().data.numpy())
        # print(yy.shape)
        # print(type(yy))
        # plt.title(label_word[new_raw_label[yy]])
        # plt.show()
        loss = loss_func(y_pre, y)
        optimizer.zero_grad()
        pre_value = torch.argmax(y_pre, 1)
        # print(pre_value.cpu().data.numpy())
        # print(y.cpu().data.numpy())
        # print(loss.cpu().data.numpy())
        step_loss_list.append(loss.cpu().data.numpy())
        plot_1_list(step_loss_list, 'step_loss', 'step_loss.png')
        if loss <= 10:
            epoch_loss += loss.cpu().data.numpy()

        loss.backward()
        optimizer.step()

        # if i % 20 == 0:
        val_acc =accuracy(dataloader=val_dataloader)
        print('val_acc = ' +str(val_acc))
        train_acc = accuracy(dataloader=train_dataloader)
        print('train_acc = ' +str(train_acc * 100)[:4] + '%')
    epoch_train_acc.append(train_acc)
    epoch_val_acc.append(val_acc)
    epoch_loss_list.append(epoch_loss / i)
    print('epoch loss = ' + str(epoch_loss/i))
    torch.save(model, 'models/densenet.pkl')
    plot_1_list(epoch_loss_list, 'loss', 'epoch_loss.png')
    plot_2_list(epoch_train_acc, epoch_val_acc, 'acc', 'acc.png')



