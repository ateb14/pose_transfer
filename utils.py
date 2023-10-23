import torch.nn.functional as F
import numpy as np
import cv2
import math
import os

def setup_logging(run_name, root_dir):
    os.makedirs(os.path.join(root_dir, "ckpts", run_name), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "models", run_name), exist_ok=True)
    os.makedirs(os.path.join(os.path.join(root_dir, "results"), run_name), exist_ok=True)
    os.makedirs(os.path.join(os.path.join(root_dir, "logs"), run_name), exist_ok=True)

def draw_pose(keypoints,H,W):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, keypoints)
    return canvas

def draw_bodypose(canvas: np.ndarray, keypoints) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape
    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]]) * float(W)
        X = np.array([keypoint1[1], keypoint2[1]]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint[0], keypoint[1]
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas

def stitch_to_square_middle(images, background_color=0):
    image_h, image_w = images.shape[:2]
    image_square_len = max(image_h, image_w)
    pad_h = (image_square_len - image_h) // 2
    pad_w = (image_square_len - image_w) // 2
    if images.ndim == 3:
        squared_images = np.ones((image_square_len, image_square_len, images.shape[-1]), dtype='uint8') * background_color
        squared_images[pad_h : pad_h+image_h, pad_w : pad_w+image_w, :] = images
    elif images.ndim == 2:
        squared_images = np.ones((image_square_len, image_square_len), dtype='uint8') * background_color
        squared_images[pad_h : pad_h+image_h, pad_w : pad_w+image_w] = images
    else:
        assert False, f'Dimension of input images is {images.ndim}, but only 3 and 2 are supported'
    return squared_images

def dialate(binary_image, kernel_size=5):
    pad = (kernel_size - 1) // 2
    binary_image = F.pad(binary_image, pad=(pad,pad,pad,pad), mode='reflect')
    dialated_img = F.max_pool2d(binary_image, kernel_size=kernel_size, stride=1, padding=0)
    return dialated_img

def erode(binary_image, kernel_size=5):
    rev_bin_img = 1 - binary_image
    return 1 - dialate(rev_bin_img, kernel_size=kernel_size)

def imopen(binary_image, kernel_size=5):
    return dialate(erode(binary_image, kernel_size), kernel_size)

def imclose(binary_image, kernel_size=5):
    return erode(dialate(binary_image, kernel_size), kernel_size)