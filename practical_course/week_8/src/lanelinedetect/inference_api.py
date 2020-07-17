from .MyDeepLab import *
from torchvision import transforms
from PIL import Image
import numpy as np


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def preprocess(img):
    """
    输入图像预处理
    """
    # Normalize
    img = np.array(img).astype(np.float32)
    img /= 255.0
    img -= (0.485, 0.456, 0.406)
    img /= (0.229, 0.224, 0.225)
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float()
    return img

def postprocess(output):
    """
    后处理：将输出结果转换为类别分割图
    """
    output = output.detach().numpy()
    output = output[0]
    output = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
    return output


def load_model_from_path(name, model_path):
    model = MyDeepLab(backbone=name, output_stride=16, num_classes=8)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['state_dict'])
    return model

def test(model, img_path):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    input_data = preprocess(img)
    output_data = model(input_data)
    output = postprocess(output_data)
    mask_img = Image.fromarray(output)
    mask_img.putpalette(get_palette(8))
    return mask_img 

if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    name = 'mobilenet'
    model_path = 'model_mobilenet.pth.tar'
    model = load_model_from_path(name=name, model_path=model_path)
    mask_img = test(model, 'test.jpg')
    mask_img.save('test_mask.png')

