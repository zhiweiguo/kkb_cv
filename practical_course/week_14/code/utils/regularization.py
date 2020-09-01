import numpy as np
import random
import cv2

import torch
import torch.optim as optim
from torchvision import datasets, transforms
import argparse as args
import torch.nn.functional as F
from torch import nn


''' Label Smooth '''
def label_smoothing(inputs, epsilon=0.1, K=2):
    #K = inputs.get_shape().as_list()[-1]    # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)

''' Dropblock '''
# The target of Dropblock is to use dropout at conv layers.
# The reason why we're not using dropout in conv layers but only use that in fc layers is because of the
# "share local info" nature of convolutions, which means if you randomly drop some activation units, those nearby
# feature elements will still cover the similar info which will deteriorate the effect of dropout.
# One way to overcome this is by using Dropblock, which can be regarded as a strcuturalized dropout. We, this time,
# merely, randomly choose ignoring seeds and dropped square patches around each seed simoutaneously. Therefore, we
# forced the model to negelect some local info and to focus on the rest.
#
""" some links """
# paper: https://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks.pdf
#
# Official: https://github.com/miguelvr/dropblock (gamma, a parameter, is approximated in this version)
# gamma: drop rate for randomly chosen seeds
#
# offical example: https://github.com/miguelvr/dropblock/blob/master/examples/resnet-cifar10.py
# (here we have linear scheduling implementation)
#
# Accurate gamma version: https://github.com/Randl/DropBlock-pytorch
#
# Here we use official implementation as example:
#
""" Dropblock impementation """
class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob      # <=>(1 - keep_prob)
        self.block_size = block_size    # default = 5 or 7

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            # approximation: _compute_gamma(x) = self.drop_prob / (self.block_size ** 2)
            # accurate gamma = _compute_gamma(x) *
            #                 (x.shape[2] * x.shape[3]) /                                       # => feat_map * feat_map
            #                 ((x.shape[2] - block_size + 1) * (x.shape[3] - block_size + 1))   # => (fm - bl + 1) ^ 2
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            # Warning：no channel 1 here.
            # E.g:
            # x = torch.randint(0, 256, (2, 3, 4, 4)).float()
            # x
            # gamma = 0.2 / (2*2)
            # gamma
            # mask = (torch.rand(x.shape[0], *x.shape[2:])<gamma).float()
            # mask
            # out = x * mask[:, None, :, :]     # this 'None' can help to "replicate on channel 1"

            # place mask on input device
            mask = mask.to(x.device)    # put to gpu or cpu

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]     # "None" replicate the mask one "channel" dim

            # scale output
            out = out * block_mask.numel() / block_mask.sum()
            # .numel: total num of elements
            # .sum: summation. here because only 0 & 1, that's the number of ones.
            # to be alongside with dropout: out = out / p
            #                        where: p = block_mask.sum() / block_mask.numel()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],        # dim from 3 to 4
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]     # drop the last column and row

        block_mask = 1 - block_mask.squeeze(1)      # reverse the dim from 4 to 3

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
""" Dropblock linear scheduler """
# similar to official paper: https://arxiv.org/pdf/1707.07012.pdf
class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1

class ResNetCustom():
    def __init__(self, block, layers, num_classes=1000, drop_prob=0., block_size=5):
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3
        )

    def forward(self, x):
        self.dropblock.step()  # increment number of iterations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        ''' here here'''
        x = self.dropblock(self.layer1(x))
        x = self.dropblock(self.layer2(x))
        ''' done '''
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x
