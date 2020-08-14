#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   coco_anno_process.py
@Time    :   2020/08/11 15:00:56
@Author  :   guo.zhiwei
@Contact :   zhiweiguo1991@163.com
@Desc    :   None
'''

# here put the import lib
import json
import cv2
import os


imgs_dir = 'D:/Data/coco/val2017/'
anno_file = 'D:/Data/coco/annotations_trainval2017/annotations/instances_val2017.json'

def read_json(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        json_file = json.loads(content)
    return json_file

# 提取原始的json 文件
def extract_json(file_path):
    anno_json = read_json(file_path)
    ext_json = {}
    ext_json['image_id'] = {}
    ext_json['categories'] = {}
    # 类别
    for i in range(len(anno_json['categories'])):
        id = anno_json['categories'][i]['id']
        ext_json['categories'][id] = anno_json['categories'][i]['name']
    # 图像整体信息
    for i in range(len(anno_json['images'])):
        image_id = anno_json['images'][i]['id']
        ext_json['image_id'][image_id] = {}
        ext_json['image_id'][image_id]['file_name'] = anno_json['images'][i]['file_name']
        ext_json['image_id'][image_id]['height'] = anno_json['images'][i]['height']
        ext_json['image_id'][image_id]['width'] = anno_json['images'][i]['width']
        ext_json['image_id'][image_id]['annotations'] = {}
    # 添加bbox及对应的类别标注信息
    for i in range(len(anno_json['annotations'])):
        image_id = anno_json['annotations'][i]['image_id']
        id = anno_json['annotations'][i]['id']
        ext_json['image_id'][image_id]['annotations'][id] = {}
        ext_json['image_id'][image_id]['annotations'][id]['bbox'] = anno_json['annotations'][i]['bbox']
        ext_json['image_id'][image_id]['annotations'][id]['category_id'] = anno_json['annotations'][i]['category_id']

    # 保存
    b = json.dumps(ext_json)
    with open('ext_val.json', 'w') as f:
        f.write(b)

    return 

# 画图保存显示
def draw_image(image_id, json_file, is_show=True, is_save=True):
    image_info_dict = json_file['image_id'][image_id]
    category_dict = json_file['categories']
    image = cv2.imread(os.path.join(imgs_dir, image_info_dict['file_name']))
    id_list = list(image_info_dict['annotations'].keys())
    color = (0, 0, 255)
    for id in id_list:
        bbox = image_info_dict['annotations'][id]['bbox']   # x1,y1,w,h
        cat_id = image_info_dict['annotations'][id]['category_id']
        category = category_dict[str(cat_id)]
        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),
            color, 2)
        cv2.putText(
            image,
            category,
            (int(bbox[0]), int(bbox[1]) - 5),
            cv2.FONT_HERSHEY_COMPLEX, 1*image.shape[0]/1024, color)
    if is_save:
        cv2.imwrite(image_info_dict['file_name'], image)
    if is_show:
        cv2.imshow('result.jpg',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    #extract_json(anno_file)
    #print('ext end')

    # 读取提取后的json文件并画图显示
    json_file = read_json('ext_val.json')
    image_id_list = list(json_file['image_id'].keys())
    #draw_image(image_id_list[500], json_file, is_show=True, is_save=True)
    draw_image('32901', json_file, is_show=True, is_save=True)
    

