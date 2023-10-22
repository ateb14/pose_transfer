import cv2
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import json
from utils import stitch_to_square_middle, draw_pose
from controlnet_aux import OpenposeDetector

class DeepFashionDataset(Dataset):
    def __init__(self, split, data_path):
        super().__init__()
        self.split = split
        self.data_path = data_path

        # load json file
        self.pairs = json.load(open(os.path.join(data_path, f'{split}_data.json'), 'r'))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        source_image_name = pair['source_image'].split('.')[0]+'.png'
        target_image_name = pair['target_image'].split('.')[0]+'.png'

        source_image = cv2.imread(os.path.join(self.data_path, f'{self.split}_lst_512_png', source_image_name))
        target_image = cv2.imread(os.path.join(self.data_path, f'{self.split}_lst_512_png', target_image_name))

        # 18 lines of pairs of 2D points, stored in a txt file
        source_pose_coordinate = np.loadtxt(os.path.join(self.data_path, 'normalized_pose_txt', source_image_name.split('.')[0]+'.txt'))
        target_pose_coordinate = np.loadtxt(os.path.join(self.data_path, 'normalized_pose_txt', target_image_name.split('.')[0]+'.txt'))
        
        # source_pose_image = draw_pose(source_pose_coordinate, source_image.shape[0], source_image.shape[1])
        # target_pose_image = draw_pose(target_pose_coordinate, target_image.shape[0], target_image.shape[1])
        source_pose_image = self.detect_pose(source_image)
        target_pose_image = self.detect_pose(target_image)

        return {
            'source_image': source_image,
            'target_image': target_image,
            'source_pose_coordinate': source_pose_coordinate,
            'target_pose_coordinate': target_pose_coordinate,
            'source_pose_image': source_pose_image,
            'target_pose_image': target_pose_image,
        }
    
    def detect_pose(self, image):
        if not hasattr(self, 'detector'):
            self.detector = OpenposeDetector.from_pretrained('/home/nfs/wyy/models/controlnet_aux')
        pose_map = self.detector(image,output_type='np')
        pose_map = cv2.resize(pose_map, (image.shape[1], image.shape[0]))
        return pose_map


if __name__ == '__main__':
    data_path = '/home/nfs/wyy/data/deepfashion/Data'
    split = 'train'
    dataset = DeepFashionDataset(split, data_path)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for i, data in enumerate(dataloader):
        source_image = data['source_image']
        target_image = data['target_image']
        source_pose_coordinate = data['source_pose_coordinate']
        target_pose_coordinate = data['target_pose_coordinate']
        source_pose_image = data['source_pose_image']
        target_pose_image = data['target_pose_image']
        print(i, source_image.shape, target_image.shape)
        print(source_pose_coordinate, target_pose_coordinate.shape)
        print(source_pose_image.shape, target_pose_image.shape)

        cv2.imwrite(f'./test/{i}_source_image.png', source_image[0].numpy())
        cv2.imwrite(f'./test/{i}_target_image.png', target_image[0].numpy())
        cv2.imwrite(f'./test/{i}_source_pose_image.png', source_pose_image[0].numpy())
        cv2.imwrite(f'./test/{i}_target_pose_image.png', target_pose_image[0].numpy())

        break
