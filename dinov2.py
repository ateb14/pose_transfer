import torch
from torchvision import transforms
from typing import Sequence
import os

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_dinov2_transform(
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.ToTensor(),
        transforms.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=False),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

def get_dinov2_model(path, version='dinov2_vitl14', pretrained=False):
    model = torch.hub.load('facebookresearch/dinov2', version, pretrained=pretrained)
    if pretrained == False:
        model_path = os.path.join(path, version + '_pretrain.pth')
        model.load_state_dict(torch.load(model_path), strict=True)
    return model


if __name__ == '__main__':
    preprocessor = get_dinov2_transform()
    model = get_dinov2_model('/home/nfs/wyy/models/dinov2','dinov2_vitg14').to('cuda')
    import numpy as np
    image = (255*np.random.rand(256, 256, 3)).astype(np.uint8)
    image = preprocessor(image).to('cuda')
    output = model(image.unsqueeze(0), is_training=True)

    print(output.keys())
    print(output['x_prenorm'].shape)
    print(output['x_norm_clstoken'].shape)
    print(output['x_norm_patchtokens'].shape)
