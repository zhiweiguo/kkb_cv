#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   nms_numpy.py
@Time    :   2020/07/22 10:09:09
@Author  :   guo.zhiwei
@Contact :   zhiweiguo1991@163.com
@Desc    :   numpy版本nms与soft nms实现
'''

# here put the import lib
import numpy as np

def nms(boxes, threshold):
    '''
    boxes    : shape=(sample_num, 5), 得到的所有边框坐标和置信度
    threshold: 设定的阈值
    '''
    # 得到坐标
    x_l = boxes[:,0]
    y_l = boxes[:,1]
    x_r = boxes[:,2]
    y_r = boxes[:,3]
    
    areas = (y_r - y_l) * (x_r - x_l)   # 计算面积   
    scores = boxes[:,4]                 # 边框置信度
    sort_idx = scores.argsort()[::-1]   # 按置信度排序(大->小)

    keep_boxes = []  # 结果

    while sort_idx.size > 0:
        i = sort_idx[0]
        keep_boxes.append(i)

        x_l_max = np.maximum(x_l[i], x_l[sort_idx[1:]])
        y_l_max = np.maximum(y_l[i], y_l[sort_idx[1:]])
        x_r_min = np.minimum(x_r[i], x_r[sort_idx[1:]])
        y_r_min = np.minimum(y_r[i], y_r[sort_idx[1:]])

        w = np.maximum(0, x_r_min-x_l_max+1)
        h = np.maximum(0, y_r_min-y_l_max+1)

        overlaps = w * h    # 交集面积
        ious = overlaps / (areas[i]+areas[sort_idx[1:]] - overlaps + 1)  #交并比

        idx = np.where(ious<=threshold)[0]   # 过滤iou大于阈值的边框

        sort_idx = sort_idx[idx+1]     # 得到剩余框中置信度最高的框

    return keep_boxes


def soft_nms(boxes, Nt, sigma=0.3, threshold=0.001):
    '''
    boxes     : shape=(sample_num, 5), 得到的所有边框坐标和置信度
    Nt        : IOU阈值门限
    sigma     : 高斯函数的方差
    threshold : 置信度得分阈值
    '''
    num = boxes.shape[0]
    indexes = np.arange(num)

    # 得到坐标
    x_l = boxes[:,0]
    y_l = boxes[:,1]
    x_r = boxes[:,2]
    y_r = boxes[:,3]

    areas = (y_r - y_l) * (x_r - x_l)   # 计算面积   
    scores = boxes[:,4]                 # 边框置信度
    sort_idx = scores.argsort()[::-1]   # 按置信度排序(大->小)

    keep_boxes = []  # 结果

    while sort_idx.size > 0:
        i = sort_idx[0]      # 将置信度最高的box索引放入最终要保留的结果中
        keep_boxes.append(i)

        x_l_max = np.maximum(x_l[i], x_l[sort_idx[1:]])
        y_l_max = np.maximum(y_l[i], y_l[sort_idx[1:]])
        x_r_min = np.minimum(x_r[i], x_r[sort_idx[1:]])
        y_r_min = np.minimum(y_r[i], y_r[sort_idx[1:]])

        w = np.maximum(0, x_r_min-x_l_max+1)
        h = np.maximum(0, y_r_min-y_l_max+1)

        overlaps = w * h    # 交集面积
        ious = overlaps / (areas[i]+areas[sort_idx[1:]] - overlaps + 1)  #交并比

        weights = np.exp(-(ious**2)/sigma)  # 计算权重
        scores[sort_idx[1:]] = weights * scores[sort_idx[1:]] # 所有边框更新score值: 权重*score
        sort_idx_w = scores[sort_idx[1:]].argsort()[::-1]  # 新的score排序
        sort_idx[1:] = sort_idx[1:][sort_idx_w]  # 索引重新排列，保持剩余的边框中score由大到小
        idx = np.where(scores[sort_idx[1:]]>=threshold)[0]   # 过滤score_w大于阈值的边框

        sort_idx = sort_idx[idx+1]     # 得到剩余框索引

    return keep_boxes


if  __name__ == '__main__':
    # nms test
    boxes = np.array([[100,100,210,210,0.72],
        [250,250,420,420,0.8],
        [220,220,320,330,0.92],
        [100,100,210,210,0.72],
        [230,240,325,330,0.81],
        [220,230,315,340,0.9]]) 
    
    threshold = 0.8
    res = nms(boxes, threshold)
    print(res)
    
    # soft nms test
    Nt = 0.7
    res = soft_nms(boxes, Nt, sigma=0.3, threshold=0.1)
    print(res)

