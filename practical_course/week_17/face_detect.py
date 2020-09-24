# encoding:utf-8
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   face_detect.py
@Time    :   2020/09/24 21:47:34
@Author  :   guo.zhiwei
@Contact :   zhiweiguo1991@163.com
@Desc    :   None
'''

# here put the import lib
import os 
import dlib 
import cv2
import numpy as np 


class FaceDetect(object):
    def __init__(self, predictor_model_path, img_size=100):
        super(FaceDetect, self).__init__()
        self.img_size = img_size
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = self.load_predictor(predictor_model_path)

    def load_predictor(self, model_path):
        return dlib.shape_predictor(model_path)

    def show_landmarks(self, img, shape, show=True, save=True):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(68):
            cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), 1,
                        cv2.LINE_AA)
            # cv2.drawKeypoints(img, (sp.part(i).x, sp.part(i).y),img, [0, 0, 255])
        title = 'img_landmarks'
        if save:
            filename = title + '.jpg'
            cv2.imwrite(filename, img)
            print("save img_landmarks.jpg ok !!!")
        # os.system("open %s"%(filename)) 
        if show:
            cv2.imshow(title, img)
            cv2.waitKey(0)
            cv2.destroyWindow(title)

    def get_landmarks(self, img_path):
        img = cv2.imread(img_path)
        dets = self.detector(img, 1)     # 检测人脸
        if len(dets) == 1:                      
            shape = self.predictor(img, dets[0])      # 关键点提取
            #print("Computing descriptor on aligned image ..")
            #人脸对齐 face alignment
            face_img = dlib.get_face_chip(img, shape, size=self.img_size)
            self.show_landmarks(img, shape, show=True, save=True)                    
        else:
            print("No face !!!")
        

if __name__ == '__main__':
    img_size = 100
    predictor_model_path = './shape_predictor_68_face_landmarks.dat'
    face_detect = FaceDetect(predictor_model_path=predictor_model_path, img_size=img_size)
    img_path = './1.jpg'
    face_detect.get_landmarks(img_path)

