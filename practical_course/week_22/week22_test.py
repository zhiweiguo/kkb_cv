#coding:utf-8
# 路径置顶
import sys
#sys.path.append("/home/aistudio/external-libraries/")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.getcwd())
from torch.nn.modules.distance import PairwiseDistance
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import time
# 导入文件
from train_dataset import TrainDataset
from dataset_lfw import TestDataset
from triplet_loss import TripletLoss
import torchvision.models as models
import torchvision.transforms as transforms
from week22_train import Resnet18Triplet
import torch
from eval_lfw_tool import *

config = {'name':'config'}
config['test_pairs_paths'] = 'test_pairs_1.npy'
config['LFW_data_path'] = 'lfw_funneled'
config['LFW_pairs'] = 'lfw-funneled/pairs.txt'
config['predicter_path'] = './shape_predictor_68_face_landmarks.dat'
config['image_size']=256
config['test_batch_size'] = 16
config['num_workers']=0

import pdb
pdb.set_trace()

# 测试数据的变换
test_data_transforms = transforms.Compose([
    # transforms.Resize([config['image_size'], config['image_size']]), # resize
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
        )
    ])
# 测试数据生成器
dataset=TestDataset(
            dir=config['LFW_data_path'],
            pairs_path=config['LFW_pairs'],
            predicter_path=config['predicter_path'],
            img_size=config['image_size'],
            transform=test_data_transforms,
            test_pairs_paths=config['test_pairs_paths']
        )
dataset_1 = dataset
test_dataloader = torch.utils.data.DataLoader(
        dataset_1,
        batch_size=config['test_batch_size'],
        num_workers=config['num_workers'],
        shuffle=False)
#for index,(img1,img2,issame) in enumerate(test_dataloader):
#    print(img1.shape)

# 模型加载
model = Resnet18Triplet(pretrained=False,embedding_dimension = 64)
if torch.cuda.is_available():
    model.cuda()
    print('Using single-gpu testing.')

model_pathi="../famous-enterprises-fr/week21/Model_training_checkpoints/model_resnet18_triplet_epoch_603.pt"
model_pathi = "outputs/resnet18_24_0.003530.pth"
model_pathi = "outputs_256/resnet18_24_0.003289.pth"
model_pathi = "outputs_64/resnet18_24_0.003392.pth"
if os.path.exists(model_pathi):
    model_state = torch.load(model_pathi)
    model.load_state_dict(model_state)          #['model_state_dict'])
    #start_epoch = model_state['epoch']
    print('loaded %s' % model_pathi)
else:
    print('不存在预训练模型！')


l2_distance = PairwiseDistance(2)
with torch.no_grad():  # 不传梯度了
    distances, labels = [], []
    progress_bar = enumerate(tqdm(test_dataloader))
    for batch_index, (data_a, data_b, label) in progress_bar:
    #for batch_index, (data_a, data_b, label) in enumerate(test_dataloader):
        # data_a, data_b, label这仨是一批的矩阵
        data_a = data_a.cuda()
        data_b = data_b.cuda()
        label = label.cuda()
        output_a, output_b = model(data_a), model(data_b)
        output_a = torch.div(output_a, torch.norm(output_a))
        output_b = torch.div(output_b, torch.norm(output_b))
        distance = l2_distance.forward(output_a, output_b)
        # 列表里套矩阵
        labels.append(label.cpu().detach().numpy())
        distances.append(distance.cpu().detach().numpy())
        #if batch_index >=3:
        #    break
    print("get all image's distance done")
    
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for distance in distances for subdist in distance])
    true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
    tar, far = evaluate_lfw(
        distances=distances,
        labels=labels,
        epoch='',
        tag='NOTMaskedLFW_aucnotmask_valid',
        version="20201102",
        pltshow=False
    )

# 打印日志内容
print('LFW_test_log:\tAUC: {:.3f}\tACC: {:.3f}+-{:.3f}\trecall: {:.3f}+-{:.3f}\tPrecision {:.3f}+-{:.3f}\t'.format(
    roc_auc,
    np.mean(accuracy),
    np.std(accuracy),
    np.mean(recall),
    np.std(recall),
    np.mean(precision),
    np.std(precision))+'\tbest_distance:{:.3f}\t'.format(np.mean(best_distances))
)
