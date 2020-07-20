import os
import torch
import torch.nn as nn
import myresnet
#import MyResNet
import mydata
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# model save
save_dir = './ckpt/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# train settings
BATCH_SIZE = 32
INPUT_SIZE = 224
LR = 0.001
EPOCH_NUM = 3
EPOCH_START = 0


# datasets
trainset = mydata.MyDataset(size=INPUT_SIZE, base_dir='dataset', split='train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=4, drop_last=False)

valset = mydata.MyDataset(size=INPUT_SIZE, base_dir='dataset', split='val')
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=4, drop_last=False)

# define model
kwargs = {}
kwargs['num_classes'] = 14
net = myresnet.get_resnet(mode = 18, use_se = False, pretrained = False, progress = True, **kwargs)
#net = MyResNet.resnet18(pretrained=False, progress=True, **kwargs)

# restore params
pretrained_model = 'resnet18-5c106cde.pth'
pre_state_dict = torch.load(pretrained_model)
pre_key_list = list(pre_state_dict.keys())
key_list = list(net.state_dict().keys())
count = 0
for key in key_list:
    if key in pre_key_list:
        if pre_state_dict[key].shape == net.state_dict()[key].shape:
            net.state_dict()[key] = deepcopy(pre_state_dict[key])
            net.state_dict()[key].requires_grad = False
            count += 1
print("恢复了{}层参数".format(count))

# define optimizer
#parameters = list(net.parameters())
parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = torch.optim.Adam(parameters, lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# cuda
if torch.cuda.is_available():
    net = net.cuda()

# loss function
loss_function = nn.CrossEntropyLoss()

max_acc = 0
train_loss_list = []
val_loss_list = []
acc_list = []
for epoch in range(EPOCH_START, EPOCH_NUM):
    print("epoch = {}/{} start train!!!".format(epoch+1, EPOCH_NUM,))
    scheduler.step()
    net.train()
    train_loss = 0.0
    for i, data in enumerate(trainloader):
        imgs, labels = data['image'], data['label']
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(imgs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print("    iter:{}/{}, train loss = {:.4f}, lr = {}"
              .format(i+1, len(trainloader), loss.item(), optimizer.state_dict()['param_groups'][0]['lr']))
    print("epoch = {}/{} train end, total train loss = {:.4f}".format(epoch+1, EPOCH_NUM, train_loss))
    train_loss_list.append(train_loss)
    # eval
    val_loss = 0.0
    right_count = 0
    val_acc = 0.0
    net.eval()
    for i, data in enumerate(valloader):
        imgs, labels = data['image'], data['label']
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        outputs = net(imgs)
        #import pdb
        #pdb.set_trace()
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        preds = outputs.argmax(1)
        right_count += preds.eq(labels).sum()
    val_acc = right_count.item() / valset.__len__() * 100
    print("epoch = {}/{}, total val loss = {}, val acc = {}\n".format(epoch+1, EPOCH_NUM, val_loss, val_acc))
    val_loss_list.append(val_loss)
    acc_list.append(val_acc)
    # model save
    if val_acc > max_acc:
        ckpt_path = os.path.join(save_dir, 'resnet18_epoch{}_{:.3f}.pth'.format(epoch+1, val_acc))
        torch.save(net.state_dict(), ckpt_path)
        max_acc = val_acc
        print("模型保存成功!!!, 保存路径为:{}\n".format(ckpt_path))
'''
import pdb
pdb.set_trace()
loss_str = ','.join(train_loss_list)
print(loss_str)
acc_str = ','.join(acc_list)
print(acc_str)
with open('result.txt', 'w') as f:
    f.write('train_loss_list {}\n'.format(','.join(train_loss_list)))
    f.write('acc_list {}\n'.format(','.join(acc_list)))
'''
'''
# plt figure
import matplotlib.pyplot as plt
# loss
x = np.array(range(EPOCH_NUM)) + 1
plt.figure()
plt.title("loss")
plt.plot(x, train_loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks(x)
#plt.ylim(ymin=0.5, ymax=3)
plt.savefig('train_loss.jpg')
#plt.show()

# acc
plt.figure()
plt.title("acc")
plt.plot(x, train_loss_list)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.xticks(x)
#plt.ylim(ymin=0.5, ymax=3)
plt.savefig('train_acc.jpg')
'''
