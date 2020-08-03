#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   softmax.py
@Time    :   2020/08/03 14:38:45
@Author  :   guo.zhiwei
@Contact :   zhiweiguo1991@163.com
@Desc    :   numpy版本与torch版本的softmax实现
'''

# here put the import lib
import numpy as np 
import torch


def softmax_np(input, dim=0):
    # numpy版本的softmax实现，支持任意>=1维度的输入
    input_shape = input.shape 
    if len(input_shape) == 0:
        return []
    elif len(input_shape) == 1:
        dim = 0 # 一维输入必须保证dim=0
        max_val = np.max(input)
    else:
        max_shape = list(input_shape)
        max_shape[dim] = 1
        max_val = np.max(input, axis=dim).reshape(max_shape)     # 求dim维度的最大值
    
    output = input - max_val
    output = np.exp(output)
    
    if len(input_shape) == 1:
        sum_val = np.sum(output)
    else:
        sum_val = np.sum(output, axis=dim).reshape(max_shape)          # (input_shape[0], 1)
    
    output /= sum_val

    return output


def softmax_torch(input, dim=0):
    # torch版本的softmax实现，支持任意>=1维度的输入
    input_shape = input.shape 
    if len(input_shape) == 0:
        return []
    elif len(input_shape) == 1:
        dim = 0 # 一维输入必须保证dim=0
        max_val, _ = torch.max(input)
    else:
        max_shape = list(input_shape)
        max_shape[dim] = 1
        max_val, _ = torch.max(input, dim=dim)
        max_val = max_val.reshape(max_shape)     # 求dim维度的最大值
    
    output = input - max_val
    output = torch.exp(output)
    
    if len(input_shape) == 1:
        sum_val = torch.sum(output, dim=dim)
    else:
        sum_val = torch.sum(output, dim=dim).reshape(max_shape)          # (input_shape[0], 1)
    
    output /= sum_val

    return output

if __name__ == '__main__':
    input = np.random.random((3,4,4))
    output = softmax_np(input, dim=2)
    print('numpy输出:\n', output)    
    print(output.sum(axis=2))
    # pytorch 
    input_torch = torch.from_numpy(input)
    output_torch = softmax_torch(input_torch, dim=2)
    print('torch输出:\n', output_torch)
    print(torch.sum(output_torch, dim=2))

    print('numpy与torch输出结果误差:\n', output-output_torch.numpy())


       