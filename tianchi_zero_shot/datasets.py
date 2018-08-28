from PIL import Image
import numpy as np
import os
import glob
import torch
import random

class ZhijiangDatasets(torch.utils.data.Dataset):
    def __init__(self, mode, root, label_to_att, name_to_label, word_vec, label_word, transform = None):
        self.label_util = 'word_embedding'
        self.root = root
        self.dir = glob.glob(root)
        self.labels = []
        self.imgs = []
        self.transform = transform
        self.label_to_att = label_to_att
        self.name_to_label = name_to_label
        self.mode = mode
        self.word_vec = word_vec
        self.label_word = label_word
#         print(self.imgs[:10])
        for img_name in self.dir:
            self.imgs.append(os.path.basename(img_name))
            if mode == 'train':
                if self.label_util == 'word_embedding':
                    self.labels.append(word_vec[label_word[name_to_label[os.path.basename(img_name)]]])
                else:
                    self.labels.append(label_to_att[name_to_label[os.path.basename(img_name)]])

    def __getitem__(self, index):
        x = self.imgs[index]
        if self.mode == 'train':
            y = self.labels[index]
        else:
            # y = 'No labels' 
            y = np.zeros((3,3))
        dir_ = os.path.dirname(self.root)
        img = Image.open(os.path.join(dir_, x)).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
#         print(dir_)
        return np.array(img), y
        
    def __len__(self):
        return len(self.imgs)


class ZhijiangDatasets2(torch.utils.data.Dataset):
    def __init__(self, mode, word_root, label_to_att, name_to_label, word_vec , transform=None):
        self.word_root = word_root
        self.dir = glob.glob(word_root)
        self.labels = []
        self.imgs = []
        self.transform = transform
        self.label_to_att = label_to_att
        self.name_to_label = name_to_label
        self.mode = mode
        self.word_vec = word_vec

        for img_name in self.dir:
            self.imgs.append(os.path.basename(img_name))
            if mode == 'train':
                self.labels.append(label_to_att[name_to_label[os.path.basename(img_name)]])

    def __getitem__(self, index):
        x = self.imgs[index]
        if self.mode == 'train':
            y = self.labels[index]
        else:
            # y = 'No labels'
            y = np.zeros((3,3))
        dir_ = os.path.dirname(self.word_root)
        img = Image.open(os.path.join(dir_, x)).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        #         print(dir_)
        return np.array(img), y

    def __len__(self):
        return len(self.imgs)



class Densenet_Datasets(torch.utils.data.Dataset):
    def __init__(self, pic_dir, name_label, mode, transformer=None):
        self.transformer = transformer
        all_dir_filenames = glob.glob(pic_dir)
        self.root = os.path.dirname(pic_dir)
        self.all_filenames = [os.path.basename(x) for x in all_dir_filenames]
        self.all_labels = [name_label[x] for x in self.all_filenames]

        num = int(len(self.all_filenames) * 0.8)
        self.train_filenames = random.sample(self.all_filenames, num)
        self.train_labels = [name_label[x] for x in self.train_filenames]
        # print(len(list(self.train_filenames)))
        self.val_filenames = list(set(self.all_filenames) - set(list(self.train_filenames)))
        self.val_labels = [name_label[x] for x in self.val_filenames]

        if mode == 'train':
            self.all_filenames = self.train_filenames
            self.all_labels = self.train_labels
        else:
            self.all_filenames = self.val_filenames
            self.all_labels = self.val_labels


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.all_filenames[index])).convert('RGB')
        if self.transformer:
            img = self.transformer(img)
        return np.array(img), int(self.all_labels[index].replace('ZJL', ''))

    def __len__(self):
        return len(self.all_filenames)