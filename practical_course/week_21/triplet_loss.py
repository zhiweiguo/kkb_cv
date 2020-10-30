import torch
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance


class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)
    def forward(self, anchor, positive, negative):
        # [在triplet_loss.py 的 13行位置处，填写代码，完成triplet loss的计算]
        # 法一:
        loss = max(self.pdist(anchor, positive)-self.pdist(anchor, negative)+self.margin, 0)
        
        return loss

