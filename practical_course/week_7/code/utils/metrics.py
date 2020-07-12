import numpy as np


class Evaluator(object):
    """
    模型评估类定义
    """
    def __init__(self, num_class):
        self.num_class = num_class  # 类别数
        self.confusion_matrix = np.zeros((self.num_class,)*2)  # 混淆矩阵

    def Pixel_Accuracy(self):
        # 像素准确率
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # 各个类别的像素精度(查准率)
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        # mIoU 各个类别的平均交并比
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # 频权交并比:根据每个类出现的频率为其设置权重
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        # 得到混淆矩阵
        mask = (gt_image >= 0) & (gt_image < self.num_class) # 保证类别标签的有效性
        # 类别数 * gt的标签号 + pred的标签号
        # 示例：假如共有10个类别, 以类别5为例,若预测正确(也是5),则对应的计算值为10*5+5=55
        #       若预测错误(假如预测为4),则对应的计算值为10*5+4=54
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        # 统计label中各个值出现的次数, minlength代表最终的长度
        # np.bincount要求数组为非负,返回的结果如下所示说明:
        # 若label=[1,2,2,3,4,5,1,4]
        # 返回为[0,2,2,1,2,1],代表label中有0个0, 2个1, 2个2, 1个3,...
        # 若minlength>label中实际出现数值的最大值,后面补零
        # 如minlength=10, 则返回[0,2,2,1,2,1,0,0,0,0]
        count = np.bincount(label, minlength=self.num_class**2)
        # reshape之后，上面数值为11,22,33,44....的都在对角线上,代表预测正确的，
        # 其他位置分别可以体现出实际标签与预测标签的关系
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # 更新混淆矩阵
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
