#coding:utf-8
import os
import sys
import argparse
from data_fusion import *
from augmentation import *
from metric import *

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    # 先准备数据： 完成FDDataset函数
    batch_size=1
    # modality = "color", "depth","ir"
    train_dataset = FDDataset(mode = 'train', image_size=112)
    train_loader  = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size = batch_size,
                                drop_last   = True,
                                num_workers = 4)
    
    # 准备模型：
    from model_baseline_Fusion import FusionNet
    net = FusionNet(num_class=2)
    
    if use_gpu:
        net = torch.nn.DataParallel(net)
        net =  net.cuda()
    #[需填空] 添加loss函数
    criterion  = torch.nn.CrossEntropyLoss()
    
    ##  其他超参数：
    
    #优化需要优化得模块
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=0.1, momentum=0.9, weight_decay=0.0005)
    batch_loss_acc = np.zeros(6, np.float32)
    for epoch in range(0, 100):        
            sum_train_loss_acc = np.zeros(6,np.float32)
            sum = 0
            optimizer.zero_grad()
            iter_smooth=99
            for idx,(input, truth) in enumerate(train_loader):
                #import pdb
                #pdb.set_trace()
            
                # one iteration update  -------------
                net.train()
                if use_gpu:
                    input = input.cuda()
                    truth = truth.cuda()

                logit,_,_ = net.forward(input)
                truth = truth.view(logit.shape[0])

                loss  = criterion(logit, truth)
                precision,_ = metric(logit, truth)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                #import pdb
                #pdb.set_trace()
                # print statistics  ------------
                batch_loss_acc[:2] = np.array(( loss.item(), precision.item(),))
                sum_train_loss_acc += batch_loss_acc
                sum += 1
                if idx%iter_smooth == 0:
                    train_loss_acc = sum_train_loss_acc/sum
                    print("epoch = %s ,iter = %s,images = %s,train_loss_acc = %s"%(epoch,idx,idx*batch_size,train_loss_acc))
                    sum = 0
            
