import os
import gzip
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing

def load_mnist(path, kind='train'):
    """
    从指定路径加载 MNIST 数据集
    """
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

# 得到训练集数据
#train_imgs, train_labels = load_mnist('./mnist', kind='train')
#test_imgs, test_labels = load_mnist('./mnist', kind='t10k')

class MnistDataSet:
    def __init__(self, data_dir, feature_dim=10, batch_size=16, shuffle=False):
        self.data_dir = data_dir
        self.feature_dim = feature_dim   # 特征维度
        self.batch_size = batch_size     # 每个step的样本数
        self.shuffle =shuffle            # 训练集是否打乱顺序
        self.train_features = None       # 训练集模型输入特征
        self.train_labels = None         # 训练集模型标签
        self.test_features = None        # 测试集模型输入特征
        self.test_labels = None          # 测试集模型标签
        self.train_kind = 'train'        # 训练集文件标识
        self.test_kind = 't10k'          # 测试集文件标识
        self.preprocess()                # 预处理，产生训练集和测试集
    
    def preprocess(self):
        # 加载数据集
        train_imgs, train_labels = load_mnist(self.data_dir, kind=self.train_kind)
        test_imgs, test_labels = load_mnist(self.data_dir, kind=self.test_kind)
        # 数据标准化
        zscore = preprocessing.StandardScaler()                # zscore标准化方式
        train_imgs = zscore.fit_transform(train_imgs)
        test_imgs = zscore.transform(test_imgs)                # 测试集与训练集保持一致的zscore方式
        # PCA降维
        pca = PCA(n_components=self.feature_dim)  
        self.train_features = pca.fit_transform(train_imgs)    # 训练集pca降维
        self.test_features = pca.transform(test_imgs)          # 测试集pca降维
        self.train_labels = train_labels
        self.test_labels = test_labels
    
    def next(self):
        """
        生产batch数据，用于训练
        """
        train_num = self.train_features.shape[0]
        idx = list(range(train_num))
        if self.shuffle:
            idx = random.shuffle(idx)

        start = 0
        while start + self.batch_size < train_num:
            batch_data = self.train_features[idx[start:start+self.batch_size], :]
            batch_label = self.train_labels[idx[start:start+self.batch_size]]
            
            yield batch_data, batch_label   # 返回batch个数据及标签
            start += self.batch_size
        # 不够一个batch时，返回剩下的所有数据
        batch_data = self.train_features[idx[start:], :]
        batch_label = self.train_labels[idx[start:]]
        return batch_data, batch_label
        

if __name__ == "__main__":

    epoch = 10
    dataset = MnistDataSet('./mnist',feature_dim=10, batch_size=16)
    idx = 0
    for batch_data, batch_label in dataset.next():
        #batch_data,  batch_label = dataset.next()
        print(np.sum(batch_data - dataset.train_features[idx:idx+16]))
        print(np.sum(batch_label - dataset.train_labels[idx:idx+16]))
        idx += 16
        if idx > 60000:
            idx = 60000
        #print(batch_data)
        #print(batch_label)