import os
import cv2
import numpy as np
from controlnet_aux import OpenposeDetector
from tqdm import tqdm


detector = OpenposeDetector.from_pretrained('/home/nfs/wyy/models/controlnet_aux').to('cuda')
os.makedirs('/home/nfs/wyy/data/deepfashion/Data/train_lst_512_pose', exist_ok=True)
os.makedirs('/home/nfs/wyy/data/deepfashion/Data/test_lst_512_pose', exist_ok=True)

def detect(path,pose_path):
    for cnt, name in tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path))):
        if cnt < 40883:
            continue
        if name.split('.')[-1] != 'png':
            continue
        image = cv2.imread(os.path.join(path, name))
        pose_map, pose = detector(image,output_type='np')
        pose_map = cv2.resize(pose_map, (image.shape[1], image.shape[0]))
        cv2.imwrite(f'{pose_path}/{name}', pose_map)
        # cv2.imwrite(f'test/{name}', pose_map)
        if len(pose) == 0:
            pose = np.zeros((18, 2), dtype=np.float32)-1
        else:
            pose = pose[0].body.keypoints
            pose = np.array([[keypoint.x, keypoint.y] if keypoint is not None else [-1, -1] for keypoint in pose])
        np.savetxt(f'{pose_path}/{name.split(".")[0]}.txt', pose)
        # np.savetxt(f'test/{name.split(".")[0]}.txt', pose)
        

detect('/home/nfs/wyy/data/deepfashion/Data/train_lst_512_png', '/home/nfs/wyy/data/deepfashion/Data/train_lst_512_pose')
detect('/home/nfs/wyy/data/deepfashion/Data/test_lst_512_png', '/home/nfs/wyy/data/deepfashion/Data/test_lst_512_pose')