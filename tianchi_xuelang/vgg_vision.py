import torch
import numpy as np
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import os
from torch import nn

os.chdir('./input')


class UNet11(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
#         self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

#         self.relu = model.features[0]
#         torch.nn.Sequential(
        self.conv0 = self.encoder[0]
        self.relu = self.encoder[1]
        self.conv1= self.encoder[2]
        self.relu = self.encoder[3]
        self.max = self.encoder[4]
        self.conv2 = self.encoder[5]
        self.relu = self.encoder[6]
        self.conv3 = self.encoder[7]
        self.relu = self.encoder[8]
        self.max = self.encoder[9]
        self.conv5 = self.encoder[10]
        self.relu = self.encoder[11]
        self.conv6 = self.encoder[12]
        self.relu = self.encoder[13]
        self.conv7 = self.encoder[14]
        self.relu = self.encoder[15]
        self.max = self.encoder[16]
        self.conv5 = self.encoder[17]
        self.relu = self.encoder[18]
        self.conv6 = self.encoder[19]
        self.relu = self.encoder[20]
        self.conv7 = self.encoder[21]
        self.relu = self.encoder[22]
        self.max = self.encoder[23]
        self.conv5 = self.encoder[24]
        self.relu = self.encoder[25]
        self.conv6 = self.encoder[26]
        self.relu = self.encoder[27]
        self.conv7 = self.encoder[28]
        self.relu = self.encoder[29]
        self.max = self.encoder[30]

    def forward(self, x):
        conv0 = self.conv0(x)
        relu0 = self.relu(conv0)
        conv1 = self.conv1
        relu1 = self.relu
        max1 = self.max
        conv2 = self.conv2
        relu2 = self.relu = [6]
        conv3 = self.conv3 = [7]
        relu3 = self.relu = [8]
        max2 = self.max = [9]
        conv5 = self.conv5 = [10]
        self.relu = [11]
        self.conv6 = [12]
        self.relu = [13]
        self.conv7 = [14]
        self.relu = [15]
        self.max = [16]
        self.conv5 = [17]
        self.relu = [18]
        self.conv6 = [19]
        self.relu = [20]
        self.conv7 = [21]
        self.relu = [22]
        self.max = [23]
        self.conv5 = [24]
        self.relu = [25]
        self.conv6 = [26]
        self.relu = [27]
        self.conv7 = [28]
        self.relu = [29]



        conv0 = self.conv0(x)
#         conv1 = self.conv1(conv0)
        conv1 = conv0
        conv2 = self.conv2(conv1)
        conv3s = self.conv3s(conv2)
        conv3 = self.conv3(conv3s)
        conv4s = self.conv4s(conv3)
        conv4 = self.conv4(conv4s)
        return conv2
#         return conv5
net = UNet11()
print(net)
ret = net(x.reshape(1, 3, 224, 224))
ret = ret.data.numpy()
print(ret[0, 4, 0, :10])
ret.shape