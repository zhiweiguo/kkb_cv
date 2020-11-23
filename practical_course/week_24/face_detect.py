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
import sys
sys.path.append('../')
import os 
import dlib 
import cv2
import numpy as np 
import torch
from collections import OrderedDict
from process.augmentation import *
import torch.nn.functional as F
from resnet_triplet import Resnet18Triplet
import torchvision.transforms as transforms


# 人脸特征提取输入数据转换
input_data_transforms = transforms.Compose([
    # transforms.Resize([config['image_size'], config['image_size']]), # resize
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
        )
    ])


# 获取不同规格的活体检测模型
def get_fas_model(model_name, num_class, is_first_bn):
    if model_name == 'baseline':
        from model.model_baseline import Net
    elif model_name == 'model_A':
        from model.FaceBagNet_model_A import Net
    elif model_name == 'model_B':
        from model.FaceBagNet_model_B import Net
    elif model_name == 'model_C':
        from model.FaceBagNet_model_C import Net

    net = Net(num_class=num_class,is_first_bn=is_first_bn)
    return net



class Face(object):
    def __init__(self, predictor_model_path, fas_model_path, fas_model_name, rec_model, img_size=100):
        super(Face, self).__init__()
        self.img_size = img_size
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = self.load_predictor(predictor_model_path)
        self.fas_model = self.load_fas_model(fas_model_name, fas_model_path)
        #self.rec_model = dlib.face_recognition_model_v1(rec_model)
        self.rec_model = self.load_rec_model('cbam_resnet18_24_0.003460.pth')  # 加载自己训练好的人脸特征提取模型
        self.face_features = self.get_face_feature_lib()

    def load_rec_model(self, model_path):
        model = Resnet18Triplet(pretrained=False,embedding_dimension = 128)
        if torch.cuda.is_available():
            model.cuda()
            print('Using single-gpu testing.')
        if os.path.exists(model_path):
            model_state = torch.load(model_path, map_location='cpu')
            model.load_state_dict(model_state)          #['model_state_dict'])
            #start_epoch = model_state['epoch']
            print('loaded %s' % model_path)
        else:
            print('不存在预训练模型！')
        return model


    def load_fas_model(self, model_name, model_path):
        fas_model = get_fas_model(model_name=model_name, num_class=2, is_first_bn=True)
        if torch.cuda.is_available():
            state_dict = torch.load(model_path, map_location='cuda')
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        fas_model.load_state_dict(new_state_dict)
        if torch.cuda.is_available():
            net = fas_model.cuda()
        return fas_model

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

    def get_landmarks(self, img):
        dets = self.detector(img, 1)     # 检测人脸
        if len(dets) == 1:                      
            shape = self.predictor(img, dets[0])      # 关键点提取
            #print("Computing descriptor on aligned image ..")
            return shape
        else:
            print("No face !!!")
            return None

    def face_alignment(self, img, landmarks):
        #人脸对齐 face alignment
        face_img = dlib.get_face_chip(img, landmarks, size=self.img_size)
        #self.show_landmarks(img, shape, show=False, save=True)          
        return face_img          

    def get_face_status(self, face_img, landmarks):
        #h, w = face_img.shape[0], face_img.shape[1]
        #landmarks = np.matrix([[p.x, p.y] for p in landmarks.parts()]) # 得到68*2的np
        #x_min, x_max = np.min(landmarks[:,0]), np.max(landmarks[:,0])
        #y_min, y_max = np.min(landmarks[:,1]), np.max(landmarks[:,1])
        #new_size = max(x_max-x_min+20, y_max-y_min+30)
        #face_img = face_img[max(0, y_min-25):min(h, y_max+5), max(0, x_min-10):min(w, x_max+10), :]
        face_img = color_augumentor(face_img, target_shape=(64, 64, 3), is_infer=True)
        num = len(face_img)
        face_img = np.concatenate(face_img, axis=0)
        face_img = np.transpose(face_img, (0, 3, 1, 2))
        face_img = face_img.astype(np.float32)
        face_img = face_img.reshape([num, 3, 64, 64])
        face_img = face_img / 255.0
        input_image = torch.FloatTensor(face_img)
        #input_image = input_image.unsqueeze(0)
        if torch.cuda.is_available():
            input_image = input_image.cuda()
        with torch.no_grad():
            logit,_,_   = self.fas_model(input_image)
            logit = logit.view(1,num,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)
            prob = F.softmax(logit, 1)
        return np.argmax(prob.detach().cpu().numpy())
    
    def compute_face_feature(self, img):
        landmarks = self.get_landmarks(img)
        face_img = self.face_alignment(img, landmarks)
        #计算对齐后人脸的128维特征向量
        face_align_input = input_data_transforms(face_img)
        face_align_input = face_align_input.unsqueeze(0)
        face_feature = self.rec_model(face_align_input) # 自己训练好的人脸特征提取模型
        face_feature = face_feature.detach().numpy()        
        
        #face_feature = self.rec_model.compute_face_descriptor(face_img)
        return np.array(face_feature)

    def get_face_feature_lib(self, img_dir='./face_lib'):
        face_features = []
        # 当前库中只有两个人
        for i in range(2):
            img_path = img_dir + '/' + str(i+1) + '.jpg'
            img = cv2.imread(img_path)
            feature = self.compute_face_feature(img)
            face_features.append(feature)
        return face_features

    def compute_distence(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return 1 - np.dot(a, b.T)/(a_norm * b_norm)

    def face_compare(self, feature):
        # 采用余弦距离计算
        dis_threshold = 0.1
        min_dist = 2
        person_idx = -1
        for i in range(len(self.face_features)):
            dist = self.compute_distence(feature, self.face_features[i])
            if dist < dis_threshold and dist < min_dist:
                min_dist = dist
                person_idx = i
        return person_idx
    
    def face_recognition(self, img):
        landmarks = self.get_landmarks(img)
        face_img = self.face_alignment(img, landmarks)
        face_status = self.get_face_status(face_img, landmarks)
        if face_status == 0:
            print('非活体人脸！！！')
            return 
        #计算对齐后人脸的128维特征向量
        face_align_input = input_data_transforms(face_img)
        face_align_input = face_align_input.unsqueeze(0)
        face_feature = self.rec_model(face_align_input) # 自己训练好的人脸特征提取模型
        face_feature = face_feature.detach().numpy()

        #face_feature = self.rec_model.compute_face_descriptor(face_img)
        face_feature = np.array(face_feature)
        person_idx = self.face_compare(face_feature)
        if person_idx >= 0:
            print("当前图像中的人是库中person_{} .".format(person_idx+1))
        else:
            print("当前图像中的人不在人脸库中!!!")


if __name__ == '__main__':
    img_size = 150
    predictor_model_path = './shape_predictor_68_face_landmarks.dat'
    fas_model_path = './global_min_acer_model.pth'
    rec_model = './dlib_face_recognition_resnet_model_v1.dat'
    face_obj = Face(predictor_model_path=predictor_model_path, 
                    fas_model_path=fas_model_path, 
                    fas_model_name='model_A', 
                    rec_model=rec_model,
                    img_size=img_size)
    img_path = './test_imgs/101.jpg'   
    img = cv2.imread(img_path)
    '''
    landmarks = face_obj.get_landmarks(img)
    face_img = face_obj.face_alignment(img, landmarks)
    face_status = face_obj.get_face_status(face_img, landmarks)
    face_res = '活体人脸' if face_status == 1 else '假体人脸'
    print(face_res)
    '''
    face_obj.face_recognition(img)

    # 调用摄像头
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        face_obj.face_recognition(frame)
        if cv2.waitKey(10) == 'q':
            break
    cap.release()
    cv2.desdroyAllWindows()


