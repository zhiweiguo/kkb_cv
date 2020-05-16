# -*- coding: utf-8 -*-

# 该文件代码是研究完官方的ResNet代码结构之后重新写了一遍，
# 完成之后通过与官方实现进行对比及调试，最终与官方实现结果一致

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from PIL import Image
from torchvision import transforms
from torchvision import models



def conv_3X3(in_channels, out_channels, stride=1, padding=1, bias=False):
    # resnet中的卷积之后都有bn操作，bias可以省去，以此节省参数量
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def conv_1X1(in_channels, out_channels, stride=1, bias=False):
    # resnet中的卷积之后都有bn操作，bias可以省去，以此节省参数量
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    # 基础block
    expansion = 1 # 经过卷积之后通道扩张的倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv_3X3(in_channels, out_channels, stride)
        self.bn1   = norm_layer(out_channels)
        self.conv2 = conv_3X3(out_channels, out_channels)
        self.bn2   = norm_layer(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.stride= stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv_1X1(in_channels, out_channels)
        self.bn1   = norm_layer(out_channels)
        self.conv2 = conv_3X3(out_channels, out_channels, stride)
        self.bn2   = norm_layer(out_channels)
        self.conv3 = conv_1X1(out_channels, out_channels*self.expansion)
        self.bn3   = norm_layer(out_channels*self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.stride= stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class MyResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, norm_layer=None):
        super(MyResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = norm_layer(self.in_channels)
        self.relu  = nn.ReLU(inplace=True)
        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1   = self._make_layer(block, 64, layers[0])
        self.layer2   = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3   = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4   = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc       = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 需要降采样
            downsample = nn.Sequential(
                conv_1X1(self.in_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample, norm_layer=norm_layer))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, norm_layer=norm_layer))
        
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    
# 函数接口定义
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = MyResNet(block, layers, **kwargs)
    if pretrained:
        print("开始下载预训练模型")
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
        print("预训练模型参数加载完成")
    return model

def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def preprocess(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    return input_batch

def inference(model, input_batch):
    model.eval()
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output.shape)
    #print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.   
    res_sftmx = torch.nn.functional.softmax(output[0], dim=0)
    #print("softmax_res:{}".format(result))
    res = res_sftmx.argmax()
    print("argmax_res:{}".format(res))

    return res, res_sftmx

def get_official_model():
    #model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet152', pretrained=True)
    model = models.resnet152(pretrained=False)      
    model.load_state_dict(torch.load('C:/Users/Administrator/.cache/torch/checkpoints/resnet152-b121ed2d.pth'))
    model.eval()
    return model

if __name__ == "__main__":
    # 加载图像数据
    # import urllib
    # url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
    # try: urllib.URLopener().retrieve(url, filename)
    # except: urllib.request.urlretrieve(url, filename)
    filename = 'C:/Users/Administrator/Desktop/dog.jpg'
    # 预处理,得到输入数据
    input_batch = preprocess(filename)
    # 创建并加载模型预训练参数
    my_resnet = resnet152(pretrained=True, progress=True)
    res, res_sftmx = inference(my_resnet, input_batch)
    print("自己实现ResNet152结果:{}".format(res))

    model = get_official_model()
    res_official, res_official_sftmx = inference(model, input_batch)
    print("官方实现ResNet152结果:{}".format(res_official))

    isEqual = torch.equal(res_sftmx, res_official_sftmx)
    print("两种实现结果是否一致:{}".format(isEqual))
