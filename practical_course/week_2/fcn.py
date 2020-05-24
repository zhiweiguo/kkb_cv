import numpy as np
import torch
import torch.nn as nn


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

vgg_cfgs = {
    'vgg_11': [[64, 'M'], [128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'vgg_13': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'vgg_16': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
    'vgg_19': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']],
}


class FCN(nn.Module):
    pretrained_model = ''

    @classmethod
    def download(cls):
        return None

    def __init__(self, vgg_cfgs, mode=8, n_class=21):
        super(FCN, self).__init__()

        self.mode = mode

        self.in_channels = 3  # 输入通道
        self.padding = 100    # 首次卷积padding

        self.vgg_stage1 = self.make_stage_layers(vgg_cfgs[0])    # 1/2
        self.vgg_stage2 = self.make_stage_layers(vgg_cfgs[1])    # 1/4
        self.vgg_stage3 = self.make_stage_layers(vgg_cfgs[2])    # 1/8
        self.vgg_stage4 = self.make_stage_layers(vgg_cfgs[3])    # 1/16
        self.vgg_stage5 = self.make_stage_layers(vgg_cfgs[4])    # 1/32

        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        #self.upscore = self.make_upscore_layers()   # upsample
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore = nn.ConvTranspose2d(
            n_class, n_class, 64, stride=32, bias=False)
        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore16 = nn.ConvTranspose2d(
            n_class, n_class, 32, stride=16, bias=False)

        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #m.weight.data.zero_()
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                #    m.bias.data.zero_()
                    nn.init.constant_(m.bias, 0)
                print("{}初始化OK!".format(m))
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
                print("{}初始化OK!".format(m))        
    
    def make_stage_layers(self, cfg, batch_norm=False):
        layers = []          
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=self.padding)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                self.in_channels = v
                self.padding = 1 

        return nn.Sequential(*layers)

    def upscores(self, x, h, mode=8):
        if mode == 8:
            h = self.upscore(h)
            upscore2 = h  # 1/16

            h = self.score_pool4(self.pool4)
            h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
            score_pool4c = h  # 1/16

            h = upscore2 + score_pool4c  # 1/16
            h = self.upscore_pool4(h)
            upscore_pool4 = h  # 1/8

            h = self.score_pool3(self.pool3)
            h = h[:, :,
                9:9 + upscore_pool4.size()[2],
                9:9 + upscore_pool4.size()[3]]
            score_pool3c = h  # 1/8

            h = upscore_pool4 + score_pool3c  # 1/8

            h = self.upscore8(h)
            h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        elif mode == 16:
            h = self.upscore2(h)
            upscore2 = h  # 1/16

            h = self.score_pool4(self.pool4)
            h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
            score_pool4c = h  # 1/16

            h = upscore2 + score_pool4c

            h = self.upscore16(h)
            h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]]
        elif mode == 32:
            h = self.upscore(h)
            h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]]

        return h

    def forward(self, x):
        h = x
        h = self.vgg_stage1(h)

        h = self.vgg_stage2(h)

        h = self.vgg_stage3(h)
        self.pool3 = h

        h = self.vgg_stage4(h)
        self.pool4 = h

        h = self.vgg_stage5(h)
        self.pool5 = h

        h = self.fc6(h)

        h = self.fc7(h)

        h = self.upscores(x, h, mode=self.mode)

        return h


def get_fcn(vgg_name, fcn_mode, pretrained_path):
    """
    根据配置产生fcn网络结构
    input:
        vgg_name:  vgg_cfgs中的可选配置
        fcn_mode:  fcn可选模式, 8; 16, 32
        pretrained_path: True or False
    output:
        fcn_net:   fcn模型
    """
    fcn_net = FCN(vgg_cfgs=vgg_name, mode=fcn_mode)
    if pretrained_path:
        state_dict = torch.load(pretrained_path)
        fcn_net.load_state_dict(state_dict)
    return fcn_net

if __name__ == "__main__":
    vgg_name = 'vgg_16'
    fcn_mode = 8
    pretrained_path = None   
    fcn_net = get_fcn(vgg_cfgs[vgg_name], fcn_mode, pretrained_path)
    fcn_net._initialize_weights()
    print("模型初始化完成")
    # 保存模型
    torch.save(fcn_net.state_dict(), 'my_params.pth')
    print("模型保存完毕！！！")
    pretrained_path = 'my_params.pth'
    fcn_test = get_fcn(vgg_cfgs[vgg_name], fcn_mode, pretrained_path)
    print("模型加载完毕！！！")

