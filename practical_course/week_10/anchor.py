#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   anchor.py
@Time    :   2020/07/29 10:36:52
@Author  :   guo.zhiwei
@Contact :   zhiweiguo1991@163.com
@Desc    :   生成anchor代码
'''

# here put the import lib
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Anchor():
    # def __init__(self, base_size=16, anchor_ratios=[0.5,1,2]):
    #     self.base_size = base_size
    #     self.anchor_ratios = anchor_ratios
    
    def _get_base_anchor(self, base_size):
        # 根据base size 产生 [xmin,ymin,xmax,ymax]
        return np.array([1, 1, base_size, base_size]) - 1

    def _xywh(self, anchor):
        # 单个anchor: xmin,ymin,xmax,ymax -> x,y,w,h   
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        xc = anchor[0] + 0.5 * (w-1)
        yc = anchor[1] + 0.5 * (h-1)
        return np.array([xc, yc, w, h])      #.reshape((1,-1))

    def ratio_enum(self, anchor, ratios):
        # 计算不同长宽比下的anchor
        xywh = self._xywh(anchor)
        area = xywh[2] * xywh[3]
        area_ratios = area / ratios
        ws = np.round(np.sqrt(area_ratios))
        hs = np.round(ws * ratios)

        num = len(ratios)
        xywh_arr = np.zeros((num, 4))
        xywh_arr[:, 0:2] = xywh[0:2]
        xywh_arr[:,2] = ws
        xywh_arr[:,3] = hs

        ratio_anchors = self._make_anchors(xywh_arr)

        return ratio_anchors

    def scale_enum(self, ratio_anchors,  scales):
        # 计算不同尺度下的anchor
        ratios_num = ratio_anchors.shape[0]
        scales_num = len(scales)
        scale_anchors = np.zeros((ratios_num*scales_num, 4))
        for i in range(scales_num):
            for j in range(ratios_num):
                xywh = self._xywh(ratio_anchors[j])
                xywh[2:] = xywh[2:] * scales[i]
                scale_anchors[i*ratios_num+j,:] = xywh
        scale_anchors = self._make_anchors(scale_anchors)
        return scale_anchors



    def _make_anchors(self, xywh):
        # 一组anchor: x,y,w,h  ->  xmin,ymin,xmax,ymax
        anchors = np.vstack((xywh[:,0]-0.5*(xywh[:,2]-1),
                             xywh[:,1]-0.5*(xywh[:,3]-1),
                             xywh[:,0]+0.5*(xywh[:,2]-1),
                             xywh[:,1]+0.5*(xywh[:,3]-1),
                           )).T
        return anchors


    def gen_anchors(self, base_size, ratios, scales):
        # 生成anchor
        base_anchor = self._get_base_anchor(base_size)  # 根据base size得到base anchor
        ratio_anchors = self.ratio_enum(base_anchor, ratios)  # 根据base anchor 得到该面积下不同ratio的anchor
        scale_anchors = self.scale_enum(ratio_anchors, scales)  # 根据每个ratio的anchor得到不同scale的anchor
        return scale_anchors


if __name__ == '__main__':
    base_size = 16
    anchor_ratios = [0.5,1,2]
    scales = [8,16,32]
    Anchor = Anchor()
    scale_anchors = Anchor.gen_anchors(base_size, anchor_ratios, scales)
    print(scale_anchors)
    '''
    实际结果：
    [[ -84.  -40.   99.   55.]
    [ -56.  -56.   71.   71.]
    [ -36.  -80.   51.   95.]
    [-176.  -88.  191.  103.]
    [-120. -120.  135.  135.]
    [ -80. -168.   95.  183.]
    [-360. -184.  375.  199.]
    [-248. -248.  263.  263.]
    [-168. -344.  183.  359.]]
    '''
    img = cv2.imread('test.jpg')
    start_x, start_y = 200, 200
    for i in range(scale_anchors.shape[0]):
        xmin, ymin, xmax, ymax = scale_anchors[i,:]
        cv2.rectangle(img, (int(xmin)+start_x,int(ymin)+start_y), (int(xmax)+start_x,int(ymax)+start_y),(255,255,0),2)
    #plt.figure()
    #plt.imshow(img)
    cv2.imwrite('test_anchor.jpg', img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    

