import torch
import torch.nn as nn

class SegmentationLosses(object):
    """
    损失函数类定义
    """
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """构建损失函数,目前先只使用交叉熵"""
        if mode == 'ce':  # 交叉熵损失
            return self.CrossEntropyLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        """
        交叉熵损失定义
        """
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

