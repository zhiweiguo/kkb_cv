import torch
import torch.nn as nn


# diceloss版本1:
def dice_loss(input,target):
    """
    对应链接: https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
    input:  shape = [Batch, nclasses, H, W],  representing log probabilities for each class
    target: 1-hot,  representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=np.unique(target.numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs=F.softmax(input)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
    

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total

# diceloss版本2:
class SoftDiceLoss(nn.module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = probs.view(num, -1)
        intersection = (m1 * m2)

        dice_coeff = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        dice_loss = 1 - dice_coeff.sum() / num
 
        return dice_loss


# diceloss版本3:
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
 
        return loss
 
class MulticlassDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(MulticlassDiceLoss, self).__init__()
        self.weight = weight
 
    def forward(self, input, target):
 
        C = target.shape[1]
 
        # if weights is None:
        #     weights = torch.ones(C) #uniform weights for all classes
 
        dice = DiceLoss()
        totalLoss = 0
        
        for i in range(C):
            diceLoss = dice(input[:,i], target[:,i])
            if self.weight is not None:
                diceLoss *= self.weight[i]
            totalLoss += diceLoss
 
        return totalLoss

class SegmentationLosses(object):
    """
    损失函数类定义
    """
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False, num_classes=8):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.num_classes = num_classes

    def build_loss(self, mode='ce'):
        """构建损失函数,目前先只使用交叉熵"""
        if mode == 'ce':  # 交叉熵损失
            return self.CrossEntropyLoss
        elif mode == 'dice':
            return self.DiceLossFun
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

    def DiceLossFun(self, logit, target):
        """
        DiceLoss定义
        """
        target_onehot = self.one_hot(target, c=self.num_classes)

        n, c, h, w = logit.size()
        criterion = MulticlassDiceLoss(weight=self.weight)

        if self.cuda:
            criterion = criterion.cuda()
            #target_onehot = target_onehot.cuda()

        loss = criterion(logit, target_onehot)
        
        if self.batch_average:
            loss /= n

        return loss


    def one_hot(self, target, c):
        """
        将target转换为one_hot形式
        """
        # target shape: [b, h, w]
        # one_hot shape: [b, c, h, w]
        size = list(target.size())    # 得到形状,并转为list形式
        target = target.view(-1)        # 展开到一维
        ones = torch.eye(self.num_classes).cuda()
        ones = ones.index_select(0, target.long())
        size.append(c)               
        onehot = ones.view(*size)     # [b, h, w, c]
        onehot = onehot.permute(0, 3, 1, 2)  # [b, c, h, w]
        
        return onehot


if __name__ == "__main__":
    import pdb
    pdb.set_trace()
    # 验证
    seg_loss = SegmentationLosses()
    criterion = seg_loss.build_loss(mode='dice')
    # 产生数据
    logits = torch.randn((32, 8, 513, 513)).long()
    target = torch.randint(0, 8, (32, 513, 513)).long()
    loss = criterion(logits, target)
    print(loss)





