#coding:utf-8
# code for week7,week7_homework_mnist_alexnet.py
# houchangligong,zhaomingming,20200625,
import torch
from torch import  nn
from itertools import product
import sys
from data import load_mnist
import cv2
import numpy as np
import time
from thop import profile


def pdb():
    #import pdb
    #pdb.set_trace()
    pass
def print_f(msg):
    print(msg)
    pass
# 完成对一个层得flops的计算：
def get_flops(layer,fea):
    #import pdb
    #pdb.set_trace()
    # 这里填入flops计算方法：

    k_size = torch.tensor(layer.weight.data.shape)
    f_size = torch.tensor(fea.shape)
    flops = (torch.prod(k_size)+k_size[0]) * torch.prod(f_size) / k_size[0]
    flops = float(flops)
    print_f(layer)
    print_f("flops=%s / %.2f M / %.2f G "%(flops,flops/(1024.**2),flops/(1024.**3)))
    return flops
# 完成对一个层得计算时间估计和flops的计算
def forword_flops(layer,fea): 
    start_time = time.clock()
    fea= torch.relu(layer(fea))
    end_time = time.clock()
    flops=get_flops(layer,fea)
    time_cost=end_time-start_time
    print_f("time cost:%s S,computer flops:%s "%(time_cost,flops/(1024.**3)/time_cost))
    return fea,time_cost,flops
def dropout(fea,flag="train"):
    pdb()
    if flag=="train":
        size= fea.shape
        a = torch.empty(size[0],size[1]).uniform_(0, 1)
        p=torch.bernoulli(a)
        fea=fea*p
    elif flag=="evluate":
        fea=fea*0.5
    return fea

def model(feature,layers):
    y=-1
    # time cost sum
    tcs=0
    # flops sum
    fls=0
    B = len(feature)
    fea=torch.tensor(feature).view(B,1,28,28).float()
    #放大到alexnet需要的尺寸
    #import pdb
    #pdb.set_trace()
    fea = nn.functional.interpolate(fea,(224,224),mode='nearest')
    #fea = nn.functional.upsample_bilinear(fea, (224,224))
    #fea= torch.rand(100,3,224,224)
    fea=torch.cat([fea,fea,fea],1)
    B = fea.shape[0]
    print_f("feature map size:[%s,%s,%s,%s]"%(fea.shape))
    start_time = time.clock()
    fea= torch.relu(layers[0](fea))
    end_time = time.clock()
    flops=get_flops(layers[0],fea)
    time_cost=end_time-start_time
    print_f("time cost:%s S"%(end_time-start_time))
    tcs+=time_cost
    fls+=flops
    
    fea= layers[1](fea)
    #fea= torch.relu(layers[2](fea))
    fea,tc,fl=forword_flops(layers[2],fea)
    tcs+=tc
    fls+=fl

    fea= layers[3](fea)
    #fea= torch.relu(layers[4](fea))
    fea,tc,fl=forword_flops(layers[4],fea)
    tcs+=tc
    fls+=fl
    #fea= torch.relu(layers[5](fea))
    fea,tc,fl=forword_flops(layers[5],fea)
    tcs+=tc
    fls+=fl
    #fea= torch.relu(layers[6](fea))
    fea,tc,fl=forword_flops(layers[6],fea)
    tcs+=tc
    fls+=fl
    print_f("sum_time_cost:%s,sum_flops:%s,computer_flops:%s G"%(tcs,fls,fls/tcs/(1024.**3)))
    fea= layers[7](fea)
    fea = fea.view(B,9216)
    fea= torch.relu(layers[8](fea))
    fea=dropout(fea)
    
    fea= torch.relu(layers[9](fea))
    fea=dropout(fea)
    output= torch.sigmoid(dropout(layers[10](fea)))
    y=output
    #pdb()
    #y=torch.softmax(output,1)
    #y = 1.0/(1.0+torch.exp(-1.*h))
    return y

def model_forward(feature,layers):
    y=-1
    # time cost sum
    tcs=0
    # flops sum
    fls=0
    B = len(feature)
    fea=torch.tensor(feature).view(B,1,28,28).float()
    #放大到alexnet需要的尺寸
    #import pdb
    #pdb.set_trace()
    fea = nn.functional.interpolate(fea,(224,224),mode='nearest')
    #fea = nn.functional.upsample_bilinear(fea, (224,224))
    #fea= torch.rand(100,3,224,224)
    fea=torch.cat([fea,fea,fea],1)
    B = fea.shape[0]
    fea= torch.relu(layers[0](fea))   
    fea= layers[1](fea)
    fea= torch.relu(layers[2](fea))
    fea= layers[3](fea)
    fea= torch.relu(layers[4](fea))
    fea= torch.relu(layers[5](fea))   
    fea= torch.relu(layers[6](fea))
    fea= layers[7](fea)
    fea = fea.view(B,9216)
    fea= torch.relu(layers[8](fea))
    fea=dropout(fea)   
    fea= torch.relu(layers[9](fea))
    fea=dropout(fea)
    #output= torch.sigmoid(dropout(layers[10](fea)))
    output= torch.sigmoid(layers[10](fea))
    y=output
    return y


def get_acc(image_data,image_label,layers,start_i,end_i):
    correct=0
    for i in range(start_i,end_i):
             #y = model(image_data[i:i+1],layers)
             y = model_forward(image_data[i:i+1],layers)
             gt = image_label[i]
             pred = torch.argmax(y).item()
             if gt==pred:
                 correct+=1
    #print("acc=%s"%(float(correct/20.0)))
    return  float(correct/float(end_i-start_i))
def train_model(image_data,image_label,layers,lr_list):
    loss_value_before=1000000000000000.
    loss_value=10000000000000.
    #import pdb
    #pdb.set_trace()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(0,50):
        lr = lr_list[int(epoch // 50)]    # 每50个epoch降一次lr
        loss_value_before=loss_value
        loss_value=0
        #print(image_label[i])
        B = len(image_data)
        #B = 80
        #y = model(image_data[0:B],layers)
        y = model_forward(image_data[0:B],layers)
        gt=torch.tensor(image_label[0:B]).view(B,1)
        # get one_hot
        gt_vector = torch.zeros(B,1000).scatter_(1,gt.long(),1)
        #import pdb
        #pdb.set_trace()
        # 关心所有值
        #loss = torch.sum((y-gt_vector).mul(y-gt_vector))
        # 优化loss，正样本接近1，负样本远离1
        #loss1 = (y-1.0).mul(y-1.0)
        #loss = loss1[0,gt]+torch.sum(1.0/(loss1[0,0:gt]))+torch.sum(1.0/(loss1[0,gt:-1]))
        #loss_value += loss.data.item()
        gt = torch.from_numpy(image_label).long()
        loss = criterion(y, gt)
        # 更新公式
        # w  = w - (y-y1)*x*lr
        loss_value = loss
        loss.backward()
        for i in [0,2,4,5,6,8,9,10]: 
            layers[i].weight.data.sub_(layers[i].weight.grad.data*lr)
            layers[i].weight.grad.data.zero_()
            layers[i].bias.data.sub_(layers[i].bias.grad.data*lr)
            layers[i].bias.grad.data.zero_()
        train_acc=get_acc(image_data,image_label,layers,0,80)
        test_acc =get_acc(image_data,image_label,layers,80,100)
        print("epoch=%s,loss=%s/%s,train/test_acc:%s/%s"%(epoch,loss_value,loss_value_before,train_acc,test_acc))
    return layers 
# week7作业：完善并打印网络各层得参数量
def print_params_num(layers):
    print(20*"*")
    params_num=0
    params_num_K=0
    params_num_M=0
    for i in [0,2,4,5,6,8,9,10]: 
        ### 这里填入每层参数量的计算方式
        w_shape = torch.tensor(layers[i].weight.data.shape)
        b_shape = torch.tensor(layers[i].bias.data.shape)
        layer_num = torch.prod(w_shape) + torch.prod(b_shape)
        # 换算为k
        layer_num_K = layer_num/1024.
        # 换算为M
        layer_num_M = layer_num_K/1024.
        print(layers[i])
        print("layer[%s] has %s / %sK / %sM params"%(i,layer_num,layer_num_K,layer_num_M))
        params_num +=layer_num
        params_num_K +=layer_num_K
        params_num_M +=layer_num_M
    print("alexnet has %s / %sK / %sM params need to train"%(params_num,params_num_K,params_num_M))
    print(20*"*")
        

if __name__=="__main__":
    # 从输入中获取学习率
    #lr = float(sys.argv[1])
    lr = [0.1,0.02,0.01,0.005,0.001,0.0001]   
    layers=[]
    # 完善alexnet的网络结构，填入其需要得参数
    # add conv1 
    # 填写输入，输出通道数
    conv1=nn.Conv2d( 3, 96, kernel_size = 11,stride=4,padding=2)
    layers.append(conv1)
    # 填写kernel_size 和stride
    pool2=nn.MaxPool2d(kernel_size=3 , stride=2 , padding=0,ceil_mode=True)
    layers.append(pool2)
    # add conv3 
    # 填写输入，输出通道数
    conv3=nn.Conv2d(  96,  256, kernel_size = 5,stride=1,padding=2)
    layers.append(conv3)
    pool4=nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    layers.append(pool4)
    # add conv5 
    # 填写输入，输出通道数
    conv5=nn.Conv2d( 256, 384, kernel_size = 3,stride=1,padding=1)
    layers.append(conv5)
    # 填写输入，输出通道数
    conv6=nn.Conv2d( 384, 384,kernel_size = 3,stride=1,padding=1)
    layers.append(conv6)
    # 填写输入，输出通道数
    conv7=nn.Conv2d( 384, 256,kernel_size = 3,stride=1,padding=1)
    layers.append(conv7)
    # 填写kernel_size 和stride
    pool8=nn.MaxPool2d(kernel_size=3 , stride=2 , padding=0)
    layers.append(pool8)
    # 填写输入，输出神经元数
    fc9 = nn.Linear( 9216, 4096)
    layers.append(fc9)
    fc10 = nn.Linear(4096, 4096)
    layers.append(fc10)
    fc11 = nn.Linear(4096, 10)
    layers.append(fc11)
    #打印出往略得参数量
    print_params_num(layers)
    # 记载数据
    # minst 2828 dataset 60000 samples
    #mndata = MNIST('../week4/mnist/python-mnist/data/')
    #image_data_all, image_label_all = mndata.load_training()
    image_data_all, image_label_all = load_mnist('../week_4/mnist', kind='train')
    image_data=image_data_all[0:100]
    image_label=image_label_all[0:100]
    # 使用未训练的模型处理数据
    y = model(image_data,layers)
    pdb()
    # 使用为训练得模型测试 
    print("初始的未训练时模型的acc=%s"%(get_acc(image_data,image_label,layers,80,100)))
    pdb()
    # 对模型进行训练：
    train_model(image_data,image_label,layers,lr)
    # 训练完成，对模型进行测试，给出测试结果：
    print("训练完成后模型的acc=%s"%(get_acc(image_data,image_label,layers,80,100)))
