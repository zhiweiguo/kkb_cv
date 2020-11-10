import torch
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance


class TripletLoss(torch.nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)
    def forward(self, anchor, positive, negative):
        # [在triplet_loss.py 的 13行位置处，填写代码，完成triplet loss的计算]
        # 法一:
        loss = torch.clamp(self.pdist.forward(anchor, positive)-self.pdist.forward(anchor, negative)+self.margin, 0.0)
        loss = torch.mean(loss)
        
        return loss

