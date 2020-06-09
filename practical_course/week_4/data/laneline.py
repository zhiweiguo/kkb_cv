import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from data import custom_transform as tr
from data import util


    

class LaneLine(Dataset):
    """
    laneline dataset
    """
    NUM_CLASSES = 8

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('laneline'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        
        self.args = args

        with open(os.path.join(self._base_dir, split+'_imgs.txt'), 'r') as f:
            self.imgs = f.read().strip('\n').split('\n')
        with open(os.path.join(self._base_dir, split+'_labels.txt'), 'r') as f:
            self.labels = f.read().strip('\n').split('\n')
    
        assert (len(self.imgs) == len(self.labels))
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.imgs)))

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.imgs[index]).convert('RGB')
        _tmp = Image.open(self.labels[index])
        _tmp = util.encode_segment_to_class_idx(np.array(_tmp))
        _target = Image.fromarray(_tmp)
        

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            #tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            #tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'LaneLine(split=' + str(self.split) + ')'


if __name__ == "__main__":
    train_dataset = LaneLine(None, base_dir='train_val', split='train') 
    data = train_dataset.__getitem__(0)
    print(train_dataset.imgs)

    val_dataset = LaneLine(None, base_dir='train_val', split='val') 