# -*- coding:utf-8 -*-

import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        return img
    
class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return img

class RandomGaussianBlur(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return img


class MyDataset(Dataset):
    """
    week_10 classification dataset
    """
    NUM_CLASSES = 14
    LABEL_NAMES = ['其它', '今麦郎', '冰露', '百岁山', '怡宝', '百事可乐', '景甜', '娃哈哈',
                   '康师傅', '苏打水', '天府可乐', '可口可乐', '农夫山泉', '恒大冰泉']

    def __init__(self,
                 size=224,
                 base_dir='./',
                 split='train',
                 ):
        """
        :param base_dir:
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._size = size
        
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        
        self.imgs = []
        self.labels = []

        with open(os.path.join(self._base_dir, '0_annotation_train.txt'), 'r', encoding='utf-8') as f:
        #with open(self._base_dir +'/' + '0_annotation_train.txt', 'r', encoding='utf-8') as f:
            f_lines = f.readlines()
            for line in f_lines:
                line_dic = json.loads(line)
                self.imgs.append(line_dic["source"][43:])  # 0-37是绝对路径
                self.labels.append(line_dic["annotation"][0]["name"])

        assert (len(self.imgs) == len(self.labels))
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.imgs)))

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img = Image.open(os.path.join(self._base_dir, self.imgs[index])).convert('RGB')
        #img = Image.open(self._base_dir + '/' + self.imgs[index]).convert('RGB')
        img = img.resize((self._size, self._size), Image.BILINEAR)
        for split in self.split:
            if split == "train":
                img = self.transform_tr(img)
            elif split == 'val':
                img = self.transform_val(img)
        sample = {'image': img, 'label': self.LABEL_NAMES.index(self.labels[index])}
        return sample


    def transform_tr(self, img):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(img)

    def transform_val(self, img):

        composed_transforms = transforms.Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(img)

    def __str__(self):
        return 'classification dataset(split=' + str(self.split) + ')'


if __name__ == "__main__":

    train_dataset = MyDataset(size=224, base_dir='dataset', split='train')
    for i in range(150):
        data = train_dataset.__getitem__(i)
        print(data['label'])

