"""The code was copied from liorshk's 'face_pytorch' repository:
    https://github.com/liorshk/facenet_pytorch/blob/master/eval_metrics.py

    Which in turn was copied from David Sandberg's 'facenet' repository:
        https://github.com/davidsandberg/facenet/blob/master/src/lfw.py#L34
        https://github.com/davidsandberg/facenet/blob/master/src/facenet.py#L424

    Modified to also compute precision and recall metrics.
"""

import numpy as np
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from scipy import interpolate
import matplotlib.pyplot as plt
import os

pwd = os.path.abspath('./')

def pltimshow(fpr, tpr, roc_auc, epoch, tag, version):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_%s_%s_%s'%(epoch, tag+str('%.3f' % roc_auc), version))
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(pwd, 'ROC_images', 'ROC_%s_%s_%s.png'%(epoch, tag+str('%.3f' % roc_auc), version)))
    # plt.show()


def evaluate_lfw(distances, labels, epoch='',tag='', version='', pltshow=True, num_folds=10, far_target=1e-3):
    """Evaluates on the Labeled Faces in the Wild dataset using KFold cross validation based on the Euclidean
    基于欧式距离的k折交叉验证
    Args:
        distances: numpy array of the pairwise distances calculated from the LFW pairs. 人脸距离数组
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not. 人脸比对标注
        num_folds (int): Number of folds for KFold cross-validation, defaults to 10 folds. 默认10折交叉验证
        far_target (float): The False Accept Rate to calculate the True Accept Rate (TAR) at,
                             defaults to 1e-3.
    Returns:
        true_positive_rate: Mean value of all true positive rates across all cross validation folds for plotting
                             the Receiver operating characteristic (ROC) curve.
        false_positive_rate: Mean value of all false positive rates across all cross validation folds for plotting
                              the Receiver operating characteristic (ROC) curve.
        accuracy: Array of accuracy values per each fold in cross validation set.
        precision: Array of precision values per each fold in cross validation set.
        recall: Array of recall values per each fold in cross validation set.
        roc_auc: Area Under the Receiver operating characteristic (ROC) metric.
        best_distances: Array of Euclidean distance values that had the best performing accuracy on the LFW dataset
                         per each fold in cross validation set.
        tar: Array that contains True Accept Rate values per each fold in cross validation set
              when far (False Accept Rate) is set to a specific value.
              FAR 固定的时候，对应的TAR的值
        far: Array that contains False accept rate values per each fold in cross validation set.

    """

    # Calculate ROC metrics
    # 因为要遍历每个阈值，所以这里生成用于遍历的阈值数组，间隔为0.01
    thresholds_roc = np.arange(min(distances)-2, max(distances)+2, 0.01)
    # 调用calculate_roc_values函数，完成roc相关参数的计算
    true_positive_rate, false_positive_rate, precision, recall, accuracy, best_distances = \
        calculate_roc_values(
            thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
        )
    # 专门写个函数来计算auc,  调用sklean.metrics中的函数
    roc_auc = auc(false_positive_rate, true_positive_rate)
    if pltshow:
        pltimshow(false_positive_rate, true_positive_rate, roc_auc, epoch, tag, version)

    # Calculate validation rate
    thresholds_val = np.arange(min(distances)-2, max(distances)+2, 0.001)
    # 计算验证集的true acceptive rate ,false acceptive rate
    tar, far = calculate_val(
        thresholds_val=thresholds_val, distances=distances, labels=labels, far_target=far_target, num_folds=num_folds
    )

    return true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances,\
        tar, far


def calculate_roc_values(thresholds, distances, labels, num_folds=10):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    true_positive_rates = np.zeros((num_folds, num_thresholds))
    false_positive_rates = np.zeros((num_folds, num_thresholds))
    precision = np.zeros(num_folds)
    recall = np.zeros(num_folds)
    accuracy = np.zeros(num_folds)
    best_distances = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #函数calculate_merics 承包了所有计算：tp,fp,precision,recall,acc 
        # Find the best distance threshold for the k-fold cross validation using the train set
        accuracies_trainset = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            _, _, _, _, accuracies_trainset[threshold_index] = calculate_metrics(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        best_threshold_index = np.argmax(accuracies_trainset)

        # Test on test set using the best distance threshold
        for threshold_index, threshold in enumerate(thresholds):
            true_positive_rates[fold_index, threshold_index], false_positive_rates[fold_index, threshold_index], _, _,\
                _ = calculate_metrics(
                    threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
                )

        _, _, precision[fold_index], recall[fold_index], accuracy[fold_index] = calculate_metrics(
            threshold=thresholds[best_threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
        )

        true_positive_rate = np.mean(true_positive_rates, 0)
        false_positive_rate = np.mean(false_positive_rates, 0)
        best_distances[fold_index] = thresholds[best_threshold_index]

    return true_positive_rate, false_positive_rate, precision, recall, accuracy, best_distances


def calculate_metrics(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    # TP,FP,TN,FN的代码解释，不要再迷路了
    # 这里算了个数，不是比率
    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # 这里的写法挺好
    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)
    
    # 精确度：模型预测为正时的正确比率
    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    # 召回率：所有应该预测正确的
    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    # 所有预测正确的
    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, precision, recall, accuracy


def calculate_val(thresholds_val, distances, labels, far_target=1e-3, num_folds=10):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds_val)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    tar = np.zeros(num_folds)
    far = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the euclidean distance threshold that gives false acceptance rate (far) = far_target
        far_train = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds_val):
            _, far_train[threshold_index] = calculate_val_far(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds_val, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        tar[fold_index], far[fold_index] = calculate_val_far(
            threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
        )

    return tar, far


def calculate_val_far(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)
    # 人脸识别中，摄像头前的人与数据库中的人脸，比对通过时，我们称为 系统接受accept了当前人脸。
    # 由此，出现了接受率这个概念：就在人站在摄像头前，能够得到系统通过的概率。 
    # 所以，这里有两个接受率：
    #      本来应该被接受的人脸，最终被系统接受的概率  true accept rate: tar
    #      本来不应该被接受的人脸，最终被系统接收的概率 false accept rate:far
    # 这是面向使用场景的评价方式。
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    num_same = np.sum(actual_issame)
    num_diff = np.sum(np.logical_not(actual_issame))

    if num_diff == 0:
        num_diff = 1
    if num_same == 0:
        return 0, 0

    tar = float(true_accept) / float(num_same)
    far = float(false_accept) / float(num_diff)

    return tar, far
