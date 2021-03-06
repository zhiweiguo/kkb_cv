import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomCrop(object):
    # 直接从原图crop
    def __init__(self, crop_size, fill=0):
        self.crop_size = crop_size   # (h, w)
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        # 判断原始图片尺寸是否满足crop_size, 若不满足, 对其扩展并填充0
        padw = self.crop_size[1] - w if w < self.crop_size[1] else 0
        padh = self.crop_size[0] - h if h < self.crop_size[0] else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # 在有效范围内随机crop,得到crop_size*crop_size的图片
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[1])
        y1 = random.randint(0, h - self.crop_size[0])
        img = img.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    # 随机尺度的裁剪,先在设定比例内随机resize,再裁剪出crop_size大小的尺寸作为输入
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size     # (h, w)
        self.crop_size = crop_size     # (h, w)
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short = min(self.base_size)
        short_size = random.randint(int(short * 0.5), int(short * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size[0] - oh if oh < self.crop_size[0] else 0
            padw = self.crop_size[1] - ow if ow < self.crop_size[1] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[1])
        y1 = random.randint(0, h - self.crop_size[0])
        img = img.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size[0]
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size[1]
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size[1]) / 2.))
        y1 = int(round((h - self.crop_size[0]) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}
