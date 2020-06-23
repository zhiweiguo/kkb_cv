import os
import random
import numpy as np
from PIL import Image

def encode_segment_to_class_idx(segment_mask):
    """
    将segment图编码为训练可用的类别索引图
    """
    segment_mask = segment_mask.astype(int)
    label_mask = np.zeros((segment_mask.shape[0], segment_mask.shape[1]), dtype=np.int16)

    # 1
    label_mask[segment_mask == 200] = 1
    label_mask[segment_mask == 204] = 1
    label_mask[segment_mask == 209] = 1
    # 2
    label_mask[segment_mask == 201] = 2
    label_mask[segment_mask == 203] = 2
    # 3
    label_mask[segment_mask == 217] = 3
    # 4
    label_mask[segment_mask == 210] = 4
    # 5
    label_mask[segment_mask == 214] = 5
    # 6
    label_mask[segment_mask == 220] = 6
    label_mask[segment_mask == 221] = 6
    label_mask[segment_mask == 222] = 6
    label_mask[segment_mask == 224] = 6
    label_mask[segment_mask == 225] = 6
    label_mask[segment_mask == 226] = 6
    # 7
    label_mask[segment_mask == 205] = 7
    label_mask[segment_mask == 227] = 7
    label_mask[segment_mask == 250] = 7

    return label_mask


def decode_class_idx_to_segment(label_mask):
    segment_mask = np.zeros((label_mask.shape[0], label_mask.shape[1]), dtype=np.uint8)
    # 1
    segment_mask[label_mask == 1] = 200
    # 2
    segment_mask[label_mask == 2] = 201
    # 3
    segment_mask[label_mask == 3] = 217    
    # 4
    segment_mask[label_mask == 4] = 210    
    # 5
    segment_mask[label_mask == 5] = 214   
    # 6
    segment_mask[label_mask == 6] = 220    
    # 7
    segment_mask[label_mask == 7] = 250    

    return segment_mask
   

def crop_resize_datasets(in_dir='', out_dir=''):
    base_img_dir_name = 'ColorImage'
    base_label_dir_name = 'Label'

    crop_h = 700
    scale = 0.5
    count = 1
    for name1 in os.listdir(os.path.join(in_dir, base_img_dir_name)):
        dir1 = os.path.join(in_dir, base_img_dir_name, name1)
        for name2 in os.listdir(dir1):
            dir2 = os.path.join(dir1, name2)
            # 判断目标路径是否存在，若不存在，则创建目录
            output_img_dir = os.path.join(out_dir, base_img_dir_name, name1, name2)
            if not os.path.exists(output_img_dir):
                os.makedirs(output_img_dir)
            output_label_dir = os.path.join(out_dir, base_label_dir_name, name1, name2)
            if not os.path.exists(output_label_dir):
                os.makedirs(output_label_dir)
            for f in os.listdir(dir2):
                # img
                img_path = os.path.join(dir2, f)
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                img = img.crop((0, crop_h, w, h))  # left, upper, right, lower
                ow, oh = int(scale * w), int(scale * h)
                img = img.resize((ow, oh), Image.BILINEAR)
                img.save(os.path.join(output_img_dir, f))
                # label
                name, ext = os.path.splitext(f)
                label_path = os.path.join(in_dir, base_label_dir_name, name1, name2, name+'_bin.png')
                label = Image.open(label_path)
                label = label.crop((0, crop_h, w, h))
                label = label.resize((ow, oh), Image.NEAREST)
                label.save(os.path.join(output_label_dir, name + '_bin.png'))
                print(count)
                count +=1
    return


def get_train_val_file(data_root='D:\Data\LaneLine', output_dir='./train_val/', rate = 0.9, shuffle=True):
    img_dir_name = 'ColorImage'
    label_dir_name = 'Label'
    imgs_list = []
    labels_list = []
    for Road_name in os.listdir(data_root):
        for Record_name in os.listdir(os.path.join(data_root, Road_name, img_dir_name)):
            for Camera_name in os.listdir(os.path.join(data_root, Road_name, img_dir_name, Record_name)):
                for img_name in os.listdir(os.path.join(data_root, Road_name, img_dir_name, Record_name, Camera_name)):
                    name, ext = os.path.splitext(img_name)
                    imgs_list.append(os.path.join(data_root, Road_name, img_dir_name, Record_name, Camera_name, img_name))
                    labels_list.append(os.path.join(data_root, Road_name, label_dir_name, Record_name, Camera_name, name+'_bin.png')) 
    '''
    # 第一版
    base_img_dir_name = os.path.join(data_root, 'ColorImage') 
    base_label_dir_name = os.path.join(data_root, 'Label')
    imgs_list = []
    labels_list = []
    for name1 in os.listdir(base_img_dir_name):
        dir1 = os.path.join(base_img_dir_name, name1)
        for name2 in os.listdir(dir1):
            dir2 = os.path.join(dir1, name2)
            for img in os.listdir(dir2):
                name, ext = os.path.splitext(img)
                imgs_list.append(os.path.join(base_img_dir_name, name1, name2, img))
                labels_list.append(os.path.join(base_label_dir_name, name1, name2, name+'_bin.png'))
    '''
    assert (len(imgs_list) == len(labels_list))
    total_num = len(imgs_list)
    idx = list(range(total_num))
    if shuffle:
        random.shuffle(idx)
    train_num = int(total_num*rate)
    #train_imgs_list = imgs_list[idx[0:train_num]]
    #train_labels_list = labels_list[idx[0:train_num]]
    #val_imgs_list = imgs_list[idx[train_num:]]
    #val_labels_list = labels_list[idx[train_num:]]

    with open(os.path.join(output_dir, 'train_imgs.txt'), 'w') as f:
        for i in range(train_num):
            f.write(imgs_list[idx[i]] + '\n')

    with open(os.path.join(output_dir, 'train_labels.txt'), 'w') as f:
        for i in range(train_num):
            f.write(labels_list[idx[i]] + '\n')

    with open(os.path.join(output_dir, 'val_imgs.txt'), 'w') as f:
        for i in range(train_num, total_num):
            f.write(imgs_list[idx[i]] + '\n')

    with open(os.path.join(output_dir, 'val_labels.txt'), 'w') as f:
        for i in range(train_num, total_num):
            f.write(labels_list[idx[i]] + '\n')

    return 


if __name__ == "__main__":
    # 划分训练验证集
    data_root = '/home/aim/WorkSpace/dataset/LaneLine/'
    get_train_val_file(data_root=data_root, rate = 0.95)

    # 裁剪和转存数据集
    #in_dir = 'D:\Data\LaneLine\Road04\ColorImage_road04'
    #out_dir = 'F:\WorkSpace\HuaweiCloud\Data\LaneLine\Road04'
    #crop_resize_datasets(in_dir=in_dir, out_dir=out_dir)
