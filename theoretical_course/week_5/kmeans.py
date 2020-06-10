import torch

class KMeans:
    def __init__(self, n_clusters=3, amx_iter=None, verbose=True, device=torch.device("cpu")):
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_sample = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x, init_random=True):
        if init_random:
            init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device=)
            self.centers = x[init_row]
        else:
            self.centers = x[init_row]
        
        while True:
            self.nearest_center(x)
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break
            self.count += 1

        self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x)
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            clusters_samples = x[mask]
            centers = torch.cat([centers, torch.mean(clusters_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def representative_sample(self):
        self.representative_sample = torch.argmin(self.dists, (0))

def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda"0)
    else:
        device = torch.device("cpu")
    return device

