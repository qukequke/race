from torchvision import models
import torch
from torch.autograd import Variable
from util import *
from datasets import Densenet_Datasets
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

mode = 'train'
epochs = 50
batch_size = 20
root_test = 'data/DatasetA_test_20180813/test/*.jpg'
root_train = 'data/DatasetA_train_20180813/train/*.jpeg'

all_label_list, name_to_label, label_to_att, word_vec, label_word, word_label = get_dict(mode=mode)


transformer = transforms.Compose([# transforms.CenterCrop(),
                                # transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                # transforms.RandomRotation()
                                ])
dataset = Densenet_Datasets(root_train, name_to_label, 'train', transformer=transformer)
dataset2 = Densenet_Datasets(root_train, name_to_label, 'val', transformer=transformer)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
val_dataloader = torch.utils.data.DataLoader(dataset2, batch_size)

# model = models.densenet121(pretrained=False)
# model.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 190))

model = models.vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(
                            torch.nn.Linear(2*2*512, 1024),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.5),
                            torch.nn.Linear(1024, 201))
                            # torch.nn.ReLU(),
                            # torch.nn.Dropout(0.5),
                            # torch.nn.Linear(4096, 300))

if torch.cuda.is_available():
    model = model.cuda()

# model.classifier = torch.nn.Sequential(torch.nn.Linear(4096, 190))

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



# y_pre = torch.randn(10, 4)
# y = torch.arange(10)
# acc = accuracy(y_pre, y)
# print(acc)

epoch_loss_list = []
epoch_train_acc = []
epoch_val_acc = []

for epoch in range(epochs):
    print('The %d epoch' % epoch)
    epoch_loss = 0
    for i, (x, y) in enumerate(train_dataloader):
        # print(x.shape)
        # print(y.shape)
        x, y = Variable(x), Variable(y)
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        y_pre = model(x)
        # print(y_pre.shape)
        loss = loss_func(y_pre, y)
        epoch_loss += loss.cpu().data.numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_acc =accuracy(dataloader=val_dataloader)
    train_acc = accuracy(dataloader=train_dataloader)
    epoch_train_acc.append(train_acc.cpu().data.numpy())
    epoch_val_acc.append(val_acc.cpu().data.numpy())
    epoch_loss_list.append(epoch_loss / i)
    print('train_acc = ' +str(train_acc.cpu().data.numpy()))
    print('val_acc = ' +str(val_acc.cpu().data.numpy()))
    print('epoch loss = ' + str(epoch_loss))
    plot_1_list(epoch_loss, 'loss', 'epoch_loss.png')
    plot_2_list(epoch_train_acc, epoch_val_acc, 'acc', 'acc.png')



