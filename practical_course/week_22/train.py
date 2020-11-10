#coding:utf-8
# 路径置顶
import sys
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

from triplet_loss import TripletLoss
import torchvision.models as models
import torchvision.transforms as transforms




# 训练数据的变换
train_data_transforms = transforms.Compose([
    # transforms.Resize([config['image_size'], config['image_size']]), # resize
    #transforms.RandomHorizontalFlip(), # 随机翻转
    transforms.ToTensor(), # 变成tensor
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# 训练数据生成器
train_dataloader = torch.utils.data.DataLoader(
    dataset=TrainDataset(
        face_dir="Datasets/vggface2_test_face_notmask",
        csv_name='Datasets/vggface2_test_face_notmask.csv',
        num_triplets=100000,
        training_triplets_path='Datasets/vggface2_test_face_notmask_triplets.npy',
        transform=train_data_transforms,
        predicter_path='shape_predictor_68_face_landmarks.dat',
        img_size=256
    ),
    batch_size=128,
    num_workers=0,
    shuffle=True
)

class Resnet18Triplet(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 128.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=128, pretrained=False):
        super(Resnet18Triplet, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        input_features_fc_layer = self.model.fc.in_features
        # Output embedding
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

    def l2_norm(self, input):
        """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization and multiplication
        by scalar (alpha)."""
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        #   Equation 9: number of classes in VGGFace2 dataset = 9131
        #   lower bound on alpha = 5, multiply alpha by 2; alpha = 10
        alpha = 10
        embedding = embedding * alpha

        return embedding

if __name__ == '__main__':
    import pdb
    #pdb.set_trace()
    #pwd = os.path.abspath('./')
    start_epoch = 0
    model = Resnet18Triplet(pretrained=False,embedding_dimension = 256)
    if torch.cuda.is_available():
        model.cuda()
        print('Using single-gpu training.')

    # loss fun
    loss_fun = TripletLoss(margin=0.5).cuda()

    def adjust_learning_rate(optimizer, epoch):
        if epoch<30:
            lr =  0.125
        elif (epoch>=30) and (epoch<60):
            lr = 0.0625
        elif (epoch >= 60) and (epoch < 90):
            lr = 0.0155
        elif (epoch >= 90) and (epoch < 120):
            lr = 0.003
        elif (epoch>=120) and (epoch<160):
            lr = 0.0001
        else:
            lr = 0.00006
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    lr = 0.125
    optimizer_model = torch.optim.Adagrad(model.parameters(), lr = lr,lr_decay=1e-4,weight_decay=0)

    # 打卡时间、epoch
    total_time_start = time.time()
    start_epoch = start_epoch
    end_epoch = start_epoch + 25
    # 导入l2计算的
    l2_distance = PairwiseDistance(2)
    # 为了打日志先预制个最低auc和最佳acc在前头
    best_roc_auc = -1
    best_accuracy = -1


    # epoch大循环
    for epoch in range(start_epoch, end_epoch):
        print("\ntraining on TrainDataset! ...")
        epoch_time_start = time.time()
        triplet_loss_sum = 0
        sample_num = 0

        model.train()  # 训练模式
        # step小循环
        progress_bar = enumerate(tqdm(train_dataloader))
        for batch_idx, batch_sample in progress_bar:     
        #for batch_idx, (batch_sample) in enumerate(train_dataloader):
            # 获取本批次的数据
            # 取出三张人脸图(batch*图)
            anc_img = batch_sample['anc_img']
            pos_img = batch_sample['pos_img']
            neg_img = batch_sample['neg_img']
            
            # 模型运算
            # 前向传播过程-拿模型分别跑三张图，生成embedding和loss（在训练阶段的输入是两张图，输出带loss，而验证阶段输入一张图，输出只有embedding）
            anc_embedding = model(anc_img.cuda())
            pos_embedding = model(pos_img.cuda())
            neg_embedding = model(neg_img.cuda())
            
            anc_embedding = torch.div(anc_embedding, torch.norm(anc_embedding))
            pos_embedding = torch.div(pos_embedding, torch.norm(pos_embedding))
            neg_embedding = torch.div(neg_embedding, torch.norm(neg_embedding))
        
            # 损失计算
            # 计算这个批次困难样本的三元损失
            # 在159行处，调用triplet_loss完成loss的计算
            triplet_loss = loss_fun.forward(anc_embedding.cpu(), pos_embedding.cpu(), neg_embedding.cpu())
            
            loss = triplet_loss

            # 反向传播过程
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

            # update the optimizer learning rate
            adjust_learning_rate(optimizer_model, epoch)

            # 计算这个epoch内的总三元损失和计算损失所用的样本个数
            triplet_loss_sum += triplet_loss.item()
            sample_num +=anc_embedding.shape[0]
        
        # 计算这个epoch里的平均损失
        avg_triplet_loss =triplet_loss_sum/sample_num
        print("avg_triplet_loss= %s"%(avg_triplet_loss))
        epoch_time_end = time.time()
        # 模型保存
        torch.save(model.state_dict(), 'outputs_256/resnet18_{}_{:3f}.pth'.format(epoch, avg_triplet_loss))
