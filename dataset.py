import cv2
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import json
from utils import stitch_to_square_middle, draw_pose
from controlnet_aux import OpenposeDetector
from torchvision import transforms

class DeepFashionDataset(Dataset):
    def __init__(self, args, split, data_path):
        super().__init__()
        self.split = split
        self.data_path = data_path
        self.args = args
        self.image_size = [512, 352]

        # load json file
        self.pairs = json.load(open(os.path.join(data_path, f'{split}_data.json'), 'r'))

        self.diffusion_transform = transforms.Compose([
            transforms.ToTensor(), # [0, 1]
            transforms.RandomResizedCrop(self.image_size, scale=(1.0, 1.0),ratio=(1., 1.), \
                interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.Normalize(mean=[0.5], std=[0.5]), # [-1, 1] following N(0,I)
        ])

        # transformation for images feeded to image encoder
        if self.args.image_encoder == 'clip':
            self.image_encoder_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize( # only resize to 224x224, for we mannually crop the foreground into a square
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073],[0.26862954, 0.26130258, 0.27577711]),
            ])
        elif self.args.image_encoder == 'dinov2':
            self.image_encoder_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            raise NotImplementedError

        self.control_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((64,44), scale=(1.0, 1.0),ratio=(1., 1.), \
                interpolation=transforms.InterpolationMode.BILINEAR, antialias=False),
        ])

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        source_image_name = pair['source_image'].split('.')[0]+'.png'
        target_image_name = pair['target_image'].split('.')[0]+'.png'

        source_image = cv2.imread(os.path.join(self.data_path, f'{self.split}_lst_512_png', source_image_name))
        assert source_image is not None, f'{os.path.join(self.data_path, f"{self.split}_lst_512_png", source_image_name)}'
        target_image = cv2.imread(os.path.join(self.data_path, f'{self.split}_lst_512_png', target_image_name))
        assert target_image is not None, f'{os.path.join(self.data_path, f"{self.split}_lst_512_png", target_image_name)}'
        
        # source_image = stitch_to_square_middle(source_image, background_color=255)
        # target_image = stitch_to_square_middle(target_image, background_color=255)

        # 18 lines of pairs of 2D points, stored in a txt file
        target_pose_coordinate = np.loadtxt(os.path.join(self.data_path, f'{self.split}_lst_512_pose', target_image_name.split('.')[0]+'.txt'))
        assert target_pose_coordinate is not None, f'{os.path.join(self.data_path, f"{self.split}_lst_512_pose", target_image_name.split(".")[0]+".txt")}'

        target_pose_image = cv2.imread(os.path.join(self.data_path, f'{self.split}_lst_512_pose', target_image_name))
        assert target_pose_image is not None, f'{os.path.join(self.data_path, f"{self.split}_lst_512_pose", target_image_name)}'
        # target_pose_image = stitch_to_square_middle(target_pose_image, background_color=0)

        return {
            'source_image': source_image,
            'target_image': target_image,
            'diffusion_target_image': self.diffusion_transform(target_image), # for diffusion
            'target_pose_coordinate': target_pose_coordinate,
            'target_pose_image': target_pose_image,
            'image_encoder_preprocessed_source_image': self.image_encoder_transform(stitch_to_square_middle(source_image)), # for image encoder
        }
    
    def detect_pose(self, image):
        if not hasattr(self, 'detector'):
            self.detector = OpenposeDetector.from_pretrained('/home/nfs/wyy/models/controlnet_aux')
        pose_map = self.detector(image,output_type='np')
        pose_map = cv2.resize(pose_map, (image.shape[1], image.shape[0]))
        return pose_map


if __name__ == '__main__':
    # parse args
    import argparse, os
    from yaml import safe_load
    from accelerate.utils import set_seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/test.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        yaml_args = safe_load(f)
    parser.set_defaults(**yaml_args)
    args = parser.parse_args()

    data_path = '/home/nfs/wyy/data/deepfashion/Data'
    split = 'train'
    dataset = DeepFashionDataset(args, split, data_path)

    # print(dataset[6304]['source_image'].shape)
    # exit(0)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=48)
    from tqdm import tqdm as tdqm

    for i, data in tdqm(enumerate(dataloader), total=len(dataloader)):
        source_image = data['source_image']
        target_image = data['target_image']
        image_encoder_preprocessed_source_image = data['image_encoder_preprocessed_source_image']
        target_pose_coordinate = data['target_pose_coordinate']
        image_encoder_preprocessed_source_image = data['image_encoder_preprocessed_source_image']
        target_pose_image = data['target_pose_image']
        print(source_image.shape,target_pose_image.shape,image_encoder_preprocessed_source_image.shape,target_pose_image.shape, image_encoder_preprocessed_source_image.shape)

        cv2.imwrite(f'./test/{i}_source_image.png', source_image[0].numpy())
        cv2.imwrite(f'./test/{i}_target_image.png', target_image[0].numpy())
        cv2.imwrite(f'./test/{i}_target_pose_image.png', target_pose_image[0].numpy())
        exit(0)

