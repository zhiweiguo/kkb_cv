# -*- coding: utf-8 -*-

# 基于pytorch官方实现的ResNet优化实现
# 增加Squeeze-and-Excitation module的实现
# 对基础模块basicblock和bottleneck加入SE
# 封装了两大类接口get_resnet()和get_resnext(), 可以通过入参创建是否带SE的各种规格resnet和resnext网络结构 

import torch
import torch.nn as nn
#from torchvision.models.utils import load_state_dict_from_url
from PIL import Image
from torchvision import transforms
from torchvision import models
import copy



def conv_3X3(in_channels, out_channels, stride=1, groups=1, dilation=1, padding=1, bias=False):
    # resnet中的卷积之后都有bn操作，bias可以省去，以此节省参数量
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def conv_1X1(in_channels, out_channels, stride=1, bias=False):
    # resnet中的卷积之后都有bn操作，bias可以省去，以此节省参数量
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


class SEModule(nn.Module):
    '''
    Squeeze-and-Excitation Module: 通道注意力机制
    '''
    def __init__(self, channels, reduction=16):
        '''
        channels:   输入通道数
        reduction:  通道降维倍数        
        '''
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = conv_1X1(in_channels=channels, out_channels=channels//reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = conv_1X1(in_channels=channels//reduction, out_channels=channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w

class BasicBlock(nn.Module):
    # 基础block
    expansion = 1 # 经过卷积之后通道扩张的倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None, use_se=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock只支持groups=1和base_width=64')
        if dilation > 1:
            raise NotImplementedError('BasicBlock不支持Dilation > 1')

        self.conv1 = conv_3X3(in_channels, out_channels, stride)
        self.bn1   = norm_layer(out_channels)
        self.conv2 = conv_3X3(out_channels, out_channels)
        self.bn2   = norm_layer(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.stride= stride
        self.downsample = downsample
        self.use_se = use_se  # 是否使用SE注意力机制
        if self.use_se:
            self.se_module = SEModule(channels=out_channels, reduction=16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
    
        if self.use_se:
            out = self.se_module(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None, use_se=False):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.)) * groups
        self.conv1 = conv_1X1(in_channels, width)
        self.bn1   = norm_layer(width)
        self.conv2 = conv_3X3(width, width, stride, groups, dilation)
        self.bn2   = norm_layer(width)
        self.conv3 = conv_1X1(width, out_channels*self.expansion)
        self.bn3   = norm_layer(out_channels*self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.stride= stride
        self.downsample = downsample
        self.use_se = use_se # 是否使用SE注意力机制
        if self.use_se:
            self.se_module = SEModule(channels=out_channels*self.expansion, reduction=16)

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

        if self.use_se:
            out = self.se_module(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class MyResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, use_se=False):
        super(MyResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_se = use_se  # 是否使用SE注意力机制
        self._norm_layer = norm_layer
        self.in_channels = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = norm_layer(self.in_channels)
        self.relu  = nn.ReLU(inplace=True)
        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1   = self._make_layer(block, 64, layers[0])
        self.layer2   = self._make_layer(block, 128, layers[1], stride=2,
                                         dilate=replace_stride_with_dilation[0])
        self.layer3   = self._make_layer(block, 256, layers[2], stride=2,
                                         dilate=replace_stride_with_dilation[1])
        self.layer4   = self._make_layer(block, 512, layers[3], stride=2,
                                         dilate=replace_stride_with_dilation[2])

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc       = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 需要降采样
            downsample = nn.Sequential(
                conv_1X1(self.in_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride,
                            downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, use_se=self.use_se))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels,
                                groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer,
                                use_se=self.use_se))
        
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

def get_resnet(mode = 152, use_se = False, pretrained = False, progress = True, **kwargs):
    '''
    :param mode: ResNet层数, 可选为： 18, 34, 50, 101
    :param use_se: 是否使用SE模块(通道注意力机制)
    :param pretrained: 是否加载预训练模型参数
    :param progress: 是否显示下载模型的进度条
    :param kwargs: 可选参数
    :return: 创建好的模型对象
    '''
    modes = [18, 34, 50, 101, 152]
    names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    blocks = [BasicBlock, BasicBlock, BottleNeck, BottleNeck, BottleNeck]
    layers = [[2, 2, 2, 2], [3, 4, 6, 3], [3, 4, 6, 3], [3, 4, 23, 3], [3, 8, 36, 3]]
    idx = modes.index(mode)
    kwargs['use_se'] = use_se

    return _resnet(names[idx], blocks[idx], layers[idx], pretrained, progress,
                   **kwargs)

def get_resnext(mode = 0, use_se = False, pretrained=False, progress=True,**kwargs):
    '''
    创建resnext模型对象
    :param mode: 0(resnext50_32X4d), 1(resnext101_32X8d)
    :param use_se: 是否使用SE模块(通道注意力机制)
    :param pretrained: 是否加载预训练模型参数
    :param progress: 是否显示下载模型的进度条
    :param kwargs: 可选参数
    :return: 创建好的模型对象
    '''
    names = ['resnext50_32X4d', 'resnext101_32X8d']
    groups = [32, 32]
    width_per_group = [4, 8]
    layers = [[3, 4, 6, 3], [3, 4, 23, 3]]
    kwargs['groups'] = groups[mode]
    kwargs['width_per_group'] = width_per_group[mode]
    kwargs['use_se'] = use_se
    return _resnet(names[mode], BottleNeck, layers[mode], pretrained, progress, **kwargs)


# __all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']


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
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = trans(input_image)
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

def get_official_model(name):
    if name == 'resnet152':
        model_path = 'resnet152-b121ed2d.pth'
        model = models.resnet152(pretrained=False)
    if name == 'resnext101_32x8d':
        model_path = 'resnext101_32x8d-8ba56ff5.pth'
        model = models.resnext101_32x8d(pretrained=False)
    #model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet152', pretrained=True)
    #model = models.resnet152(pretrained=False)
    model.load_state_dict(torch.load('C:/Users/Administrator/.cache/torch/checkpoints/'+model_path, map_location='cpu'))
    model.eval()
    return model

if __name__ == "__main__":
    # 加载图像数据
    # import urllib
    # url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
    # try: urllib.URLopener().retrieve(url, filename)
    # except: urllib.request.urlretrieve(url, filename)
    filename = 'dog.jpg'
    # 预处理,得到输入数据
    input_batch = preprocess(filename)
    '''
    # 创建并加载模型预训练参数
    my_resnet = resnet152(pretrained=True, progress=True)
    res, res_sftmx = inference(my_resnet, input_batch)
    print("自己实现ResNet152结果:{}".format(res))

    model = get_official_model('resnet152')
    res_official, res_official_sftmx = inference(model, input_batch)
    print("官方实现ResNet152结果:{}".format(res_official))

    isEqual = torch.equal(res_sftmx, res_official_sftmx)
    print("两种实现结果是否一致:{}".format(isEqual))
    '''
    ######################'resnext101_32x8d'############################
    model = get_official_model('resnext101_32x8d')
    res_official, res_official_sftmx = inference(model, input_batch)
    print("官方实现resnext101_32x8d结果:{}".format(res_official))

    my_resnext = get_resnext(mode=0, use_se=True, pretrained=False, progress=True)
    model_path = 'seresnext50_32x4d-0521-b0ce2520.pth'
    #model_path = 'resnext101_32x8d-8ba56ff5.pth'
    #base_path = 'C:/Users/Administrator/.cache/torch/checkpoints/'
    state_dict = torch.load(model_path, map_location='cpu')

    my_dict = my_resnext.state_dict()
    count = 0
    for my_key, key in zip(list(my_dict.keys()), list(state_dict.keys())):
        if my_dict[my_key].shape != state_dict[key].shape:
            print("自己模型层名称:{},参数形状:{}".format(my_key, my_dict[my_key].shape))
            print("开源模型层名称:{},参数形状:{}".format(key, state_dict[key].shape))
            count += 1
        else:
            my_dict[my_key] = copy.deepcopy(state_dict[key])
    print("共有{}层参数形状不一致".format(count))
    my_resnext.load_state_dict(my_dict, strict=False)
    #my_resnext.load_state_dict(state_dict, strict=False)
    res, res_sftmx = inference(my_resnext, input_batch)
    print("自己实现resnext101_32x8d结果:{}".format(res))

    isEqual = torch.equal(res_sftmx, res_official_sftmx)
    print("两种实现结果是否一致:{}".format(isEqual))

    #from thop import profile

    # inputs = torch.cat([input_batch, input_batch, input_batch], 0)
    # flops, params = profile(my_resnext, inputs=inputs)
    # print(flops)
    # print(params)
