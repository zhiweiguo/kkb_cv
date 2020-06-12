import numpy as np
import random
import torch

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, init_random=True, device=torch.device("cpu")):
        self.n_clusters = n_clusters    # 聚类个数
        self.labels = None              # 样本对应的类别标签
        self.dists = None               # 样本与各个聚类中心点的距离
        self.centers = None             # 聚类中心点
        self.centers_move = torch.Tensor([float("Inf")]).to(device)   # 聚类中心点偏移量
        self.init_random = init_random  # 初始聚类中心点选取方式，True：kmeans, False: kmeans++ 
        self.max_iter = max_iter        # 最大迭代次数(大于该值就停止迭代)
        self.count = 0                  # 迭代次数记录
        self.device = device            # 设备 gpu/cpu
        self.threshold = 1e-3           # 偏移量阈值(小于该值就停止迭代)

    def fit(self, x):        
        self.get_init_centers(x)     # 初始化聚类中心点
        # 不满足迭代次数且聚类中心点偏移程度大于阈值时，就不断循环更新
        while self.count < self.max_iter and self.centers_move > self.threshold:
            self.nearest_center(x)   # 找到所有样本对应的距离最近的的聚类中心点
            self.update_center(x)    # 更新聚类中心点
            print("聚类中心点偏移程度:{}".format(self.centers_move))
            self.count += 1

    def get_init_centers(self, x):
        # 聚类中心点随机初始化，代表kmeans
        if self.init_random:
            init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
            self.centers = x[init_row]
        # 聚类中心点按照距离选出，代表kmeans++
        else:
            centers = torch.empty((self.n_clusters, x.shape[1])).to(self.device)
            # 随机选取一个点作为第一个聚类中心点
            idx = torch.randint(0, x.shape[0], (1,)).to(self.device)
            centers[0] = x[idx]
            # 选出剩余的聚类中心点
            dists = torch.empty((x.shape[0], self.n_clusters)).to(self.device) # 距离记录           
            for i in range(0, self.n_clusters-1):                               
                dists[:, i] = torch.norm(x - centers[i], dim=1)
                if i == 0:
                    # 找第2个聚类中心点时，只有一列距离(即与第一个中心点的距离)可参考
                    dist_max_idx = torch.argmax(dists[:, i])
                else:
                    # 各样本离各聚类中心点的最小值,min/max返回结果(val,idx)
                    dist_min = torch.min(dists[:, :i+1], axis=1) 
                    # 找出距离所有聚类中心最远的样本索引,使用最小值，所以dist_min[0],若使用最小值对应的索引，则dist_min[1]
                    dist_max_idx = torch.argmax(dist_min[0]) 
                centers[i+1] = x[dist_max_idx]  # 添加聚类中心点
            self.centers = centers

    def nearest_center(self, x):
        # 计算每个样本距离所有聚类中心点的距离，并找到距离最近的中心点，作为该样本的label
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((x.shape[0], self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.norm(sample - self.centers, dim=1)
            labels[i] = torch.argmin(dist)
            dists[i] = dist
        self.labels = labels
        self.dists = dists


    def update_center(self, x):
        # 根据各个样本对应的label，更新聚类中心点的坐标(各类别内所有样本求均值)
        centers = torch.empty((self.n_clusters, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            clusters_samples = x[mask]
            #centers = torch.cat([centers, torch.mean(clusters_samples, (0)).unsqueeze(0)], (0))
            centers[i] = torch.mean(clusters_samples, axis=0)
        self.centers_move = torch.norm(centers - self.centers)    # 各聚类中心较上次偏移的程度
        self.centers = centers


    def predict(self, x):
        # 预测样本
        n_sample = x.shape[0]
        dist_test = torch.zeros((n_sample, self.n_clusters))

        for i in range(self.n_clusters):
            dist_test[:, i] = torch.norm(x - self.centers[i,:], dim=1)
        clus_pred = torch.argmin(dist_test, axis=1)
        return clus_pred


def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda0")
    else:
        device = torch.device("cpu")
    return device

if __name__ == "__main__":
    # gen data
    data_1 = np.random.randn(200, 2) + [1, 1]
    data_2 = np.random.randn(200, 2) + [4, 4]
    data_3 = np.random.randn(200, 2) + [7, 1]
    data = np.concatenate((data_1, data_2, data_3), axis=0)
    # random split train and test dataset
    total_num = data.shape[0]
    idx = list(range(total_num))
    random.shuffle(idx)
    train_num = int(total_num * 0.8)
    train_data = data[idx[:train_num], :]
    test_data = data[idx[train_num:], :]

    kmeans = KMeans(n_clusters=3, init_random=False)
    kmeans.fit(torch.from_numpy(train_data))
    test_pred = kmeans.predict(torch.from_numpy(test_data))
    print(test_pred)
    test_data_2 = np.random.randn(200, 2) + [7, 1]
    test_pred_2 = kmeans.predict(torch.from_numpy(test_data_2))
    print(test_pred_2)
    test_pred_3 = kmeans.predict(torch.from_numpy(data_3))
    print(test_pred_3)

    


