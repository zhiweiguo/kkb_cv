#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gradient_descend.py
@Time    :   2020/09/24 22:38:13
@Author  :   guo.zhiwei
@Contact :   zhiweiguo1991@163.com
@Desc    :   None
'''

# here put the import lib
import torch
from torch.autograd import Variable


######################### 非pytorch方式梯度下降求最小值 ################################
def get_fun_val(x):
    return x**2

def get_gradient(x):
    return 2*x

x = 100
lr = 0.001
e = 1e-8
while True:
    gradient = get_gradient(x)
    x_pre = x
    y_pre = get_fun_val(x_pre)
    x = x - gradient*lr
    y = get_fun_val(x)
    loss = abs(y - y_pre)
    if loss < e:
        print("x val is {}".format(x))
        break


############################# pytorch方式梯度下降求最小值 ###############################
def loss_fun(w):
    return (w[0]*5 + w[1]*3 - 1)**2 + ((-3)*w[0] -4*w[1] + 1)**2


w = Variable(torch.tensor([1,1]).float(), requires_grad=True)
lr = 0.001
e = 1e-8
while True:
    loss = loss_fun(w)
    loss.backward()
    w.data = w.data - w.grad.data*lr
    w.grad.fill_(0)
    # w.grad.data = torch.zeros([0,0])
    print("loss : {}".format(loss))
    if loss < e:
        print("w val is {}".format(w))
        break


