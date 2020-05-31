import torch
import torch.nn as nn
import torch.nn.functional as F 
from aspp import build_aspp
from decoder import build_decoder
import mymobilenet  

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'mobilenet':
        return mymobilenet.MyMobileNetV2(output_stride, BatchNorm)
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
    

if __name__ == "__main__":
    model = MyDeepLab(backbone='mobilenet', output_stride=16)
    state_dict = torch.load('deeplab-mobilenet.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
