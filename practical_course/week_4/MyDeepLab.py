import torch
import torch.nn as nn
import torch.nn.functional as F 
from myaspp import build_aspp
from mydecoder import build_decoder
from mymobilenet import MyMobileNetV2


def build_backbone(backbone, output_stride, BatchNorm):
    # 暂时只实现了MobileNet
    if backbone == 'mobilenet':
        return MyMobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError

class MyDeepLab(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=16, num_classes=21, freeze_bn=False):
        super(MyDeepLab, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        
        return x

    def freeze_bn(self):
        for m in self.modules():            
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p    

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    model = MyDeepLab(backbone='mobilenet', output_stride=16)
    state_dict = torch.load('deeplab-mobilenet.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.size())
