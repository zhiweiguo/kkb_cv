#coding:utf-8
'''
Training an image classifier
训练一个分类器
We will do the following steps in order:
完成一下几个步骤：
1. Load and normalizing the CIFAR10 training and test datasets using torchvision
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the networok on the test data

1. 数据处理：下载，归一化，准备训练数据与测试数据
2. 定义网络模型
3. 定义损失函数
4. 在训练数据上训练
5. 在测试数据上测试


1. Loading and normalizing CIFAR10

Using torchvision, it’s extremely easy to load CIFAR10.
The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1]. .. note:

If running on Windows and you get a BrokenPipeError, try setting
the num_worker of torch.utils.data.DataLoader() to 0.
'''
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
'''
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified

Let us show some of the training images, for fun.
'''
#import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

## show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''
Out:

plane  bird  frog   cat
'''
import pdb
pdb.set_trace()
'''
2. Define a Convolutional Neural Network

Copy the neural network from the Neural Networks section before and modify it to take 3-channel images (instead of 1-channel images as it was defined).
'''

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


'''
3. Define a Loss function and optimizer

Let’s use a Classification Cross-Entropy loss and SGD with momentum.
'''
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
'''
4. Train the network

This is when things start to get interesting. We simply have to loop over our data iterator, and feed the inputs to the network and optimize.f
'''

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
'''
Out:

[1,  2000] loss: 2.251
[1,  4000] loss: 1.919
[1,  6000] loss: 1.718
[1,  8000] loss: 1.604
[1, 10000] loss: 1.511
[1, 12000] loss: 1.464
[2,  2000] loss: 1.398
[2,  4000] loss: 1.377
[2,  6000] loss: 1.362
[2,  8000] loss: 1.332
[2, 10000] loss: 1.296
[2, 12000] loss: 1.279
Finished Training
Let’s quickly save our trained model:
'''

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

'''
See here for more details on saving PyTorch models.
https://pytorch.org/docs/stable/notes/serialization.html

5. Test the network on the test data

We have trained the network for 2 passes(epoch) over the training dataset. But we need to check if the network has learnt anything at all.

We will check this by predicting the class label that the neural network outputs, 
and checking it against the ground-truth. 
If the prediction is correct, we add the sample to the list of correct predictions.

Okay, first step. Let us display an image from the test set to get familiar.
Next, let’s load back in our saved model 
(note: saving and re-loading the model wasn’t necessary here, we only did it to illustrate how to do so):
'''

net = Net()
net.load_state_dict(torch.load(PATH))

"Okay, now let us see what the neural network thinks these examples above are:"

outputs = net(images)
"The outputs are energies for the 10 classes. 
The higher the energy for a class, 
the more the network thinks that the image is of the particular class. 
So, let’s get the index of the highest energy:"
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
'''
Out:

Predicted:    cat  ship  ship  ship
The results seem pretty good.

Let us look at how the network performs on the whole dataset.
'''

#在整个数据集上测试
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
'''
Out:

Accuracy of the network on the 10000 test images: 55 %
That looks way better than chance, which is 10% accuracy (randomly picking a class out of 10 classes). Seems like the network learnt something.

Hmmm, what are the classes that performed well, and the classes that did not perform well:
哪些类识别较好？那些类识别较差？
'''

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

'''
Out:

Accuracy of plane : 50 %
Accuracy of   car : 77 %
Accuracy of  bird : 40 %
Accuracy of   cat : 28 %
Accuracy of  deer : 36 %
Accuracy of   dog : 58 %
Accuracy of  frog : 68 %
Accuracy of horse : 63 %
Accuracy of  ship : 71 %
Accuracy of truck : 59 %

Okay, so what next?

How do we run these neural networks on the GPU?

Training on GPU

Just like how you transfer a Tensor onto the GPU, you transfer the neural net onto the GPU.

Let’s first define our device as the first visible cuda device if we have CUDA available:
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
'''
Out:

cuda:0

The rest of this section assumes that device is a CUDA device.

Then these methods will recursively go over all modules and convert their parameters and buffers to CUDA tensors:
'''
net.to(device)
'''
Remember that you will have to send the inputs and targets at every step to the GPU too:
'''
inputs, labels = data[0].to(device), data[1].to(device)
'''
Why dont I notice MASSIVE speedup compared to CPU? Because your network is really small.

Exercise: Try increasing the width of your network (argument 2 of the first nn.Conv2d, and argument 1 of the second nn.Conv2d – they need to be the same number), see what kind of speedup you get.

Goals achieved:

Understanding PyTorch’s Tensor library and neural networks at a high level.
Train a small neural network to classify images


# how  to  Training on multiple GPUs

If you want to see even more MASSIVE speedup using all of your GPUs, please check out Optional: Data Parallelism.
https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
'''
