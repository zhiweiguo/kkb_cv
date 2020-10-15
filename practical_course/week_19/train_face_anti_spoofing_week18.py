import os
# os.environ['CUDA_VISIBLE_DEVICES'] =  '4,5,6,7' #'3,2,1,0'
#os.environ["PATH"] += os.pathsep + "/home/ma-user/work/face_anti/CVPR19-Face-Anti-spoofing/process"
import sys
#sys.path.append("..")
#sys.path.append("./model/backbone")
sys.path.append("./process")
#sys.path.append("./model")
#sys.path.append("./model_fusion")



import argparse
#from process.data import *
from process.data_fusion import *
from process.augmentation import *
from metric import *

if __name__ == '__main__':
    # 先准备数据：
    batch_size=128
    # modality = "color", "depth","ir"
    train_dataset = FDDataset(mode = 'train', modality="ir",image_size=112,fold_index=-1,augment = color_augumentor)
    train_loader  = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size = batch_size,
                                drop_last   = True,
                                num_workers = 4)
    
    # 准备模型：
    from model.model_baseline import Net
    #net = Net(num_class=2,is_first_bn=True)
    
    from model_fusion.model_baseline_SEFusion import FusionNet
    net = FusionNet(num_class=2)
    
    net = torch.nn.DataParallel(net)
    net =  net.cuda()
    # 准备loss
    criterion  = softmax_cross_entropy_criterion
    
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
            
    