import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.grid import Grid

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=1, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.grid = Grid(True,True,1,0,0.5,1,1.0)

    def __getitem__(self, index):
        #import pdb
        #pdb.set_trace()
        if self.augment == 0:   # 不增强        
            img, targets = self.load_img_target(index)
        elif self.augment == 1: # 随机水平翻转
            img, targets = self.load_img_target(index)
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        elif self.augment == 2: # GridMask
            img, targets = self.load_img_target(index)
            if np.random.random() < 0.5:
                img, targets = self.grid.__call__(img, targets) # GridMask
        elif self.augment == 3: # Mosaic
            if np.random.random() < 0.5:
                img, targets = self.load_mosaic(index)
            else:
                img, targets = self.load_img_target(index)
        else:                   # 全部
            if np.random.random() < 0.5:
                img, targets = self.load_mosaic(index)
            else:
                img, targets = self.load_img_target(index)
            if np.random.random() < 0.5:
                img, targets = self.grid.__call__(img.float(), targets) # GridMask
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img.float(), targets)

        return img.float(), targets



    def load_img_target(self, index):
        #import pdb 
        #pdb.set_trace()
        #img, labels = self.load_mosaic(index)
        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        print(img_path)
        # 添加
        #base_dir = '/home/aim/WorkSpace/my_workspace/PyTorch-YOLOv3/data/face_mask/JPEGImages'
        #img_path = os.path.join(base_dir, img_path)

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # boxes[:, 0] => class id
            # Returns (xc, yc, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w        # newer center x
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h        # newer center y
            boxes[:, 3] *= w_factor / padded_w              # newer width
            boxes[:, 4] *= h_factor / padded_h              # newer height

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        # Apply augmentations
        #if self.augment:
        #    if np.random.random() < 0.5:
        #        img, targets = horisontal_flip(img, targets)

        return img, targets

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets

    def __len__(self):
        return len(self.img_files)-1
        #return 128
        
    def load_mosaic(self, index):
        # loads images in a mosaic

        labels4 = []
        s = self.img_size
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
        xc, yc = s, s
        indices = [index] + [random.randint(0, len(self.label_files) - 1) for _ in range(3)]  # 3 additional image indices
        # 在可允许的范围之内，随机抽取4个indices （batchsize=16, [1, 0, 2, 4], [2, 0, 1, 4]）
        for i, index in enumerate(indices):
            # Load image
            #img, target = self.load_img_target(index)
            #img = cv2.resize(img, (s, s))
            #img_path = self.img_files[index % len(self.img_files)].rstrip()
            #img = cv2.imread(img_path)
            #(h, w) = img.shape[:2]
            img, _, (h, w) = self.load_image(index)

            label_path = self.label_files[index % len(self.img_files)].rstrip()
            target = np.loadtxt(label_path).reshape(-1, 5)
            # place img in img4
            if i == 0:  # top left
                # 把新图像先设置成原来的4倍，到时候再resize回去，114是gray
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (new/large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (original/small image)
                # 回看ppt讲解
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b # 有时边上还是灰的
            padh = y1a - y1b

            # Labels
            #x = self.label_files[index]
            x = target.copy()
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                # 此时x是0-1，同时，label是[bbox_xc, bbox_yc, bbox_w, bbox_c]
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            # a = np.array([[1, 2], [3, 4]])
            # c = np.concatenate(a, axis=0)
            # c: [1, 2, 3, 4]
            labels4 = np.concatenate(labels4, 0)    # 0是dimension
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        labels4 = torch.from_numpy(labels4)
        
        targets = torch.zeros((labels4.shape[0], 6))
        targets[:,1] = labels4[:,0]
        targets[:,2] = (labels4[:,1] + labels4[:,3])/(4*s) # x
        targets[:,3] = (labels4[:,2] + labels4[:,4])/(4*s) # y
        targets[:,4] = (labels4[:,3] - labels4[:,1])/(4*s) # w
        targets[:,5] = (labels4[:,4] - labels4[:,2])/(4*s) # h

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        # img4, labels4 = random_affine(img4, labels4,
        #                             degrees=self.hyp['degrees'],
        #                             translate=self.hyp['translate'],
        #                             scale=self.hyp['scale'],
        #                             shear=self.hyp['shear'],
        #                             border=-s // 2)  # border to remove

        img4 = torch.from_numpy(np.transpose(img4, (2,0,1)))

        return img4, targets


    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        path = self.img_files[index % len(self.img_files)].rstrip()
        print(path)
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

