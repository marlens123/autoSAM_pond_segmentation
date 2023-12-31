import albumentations as A
import numpy as np
import random
import cv2
from sklearn.utils import class_weight
import torch

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def get_training_augmentation(im_size=480, augment_mode='1'):
    """
    structure inspired by https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
    
    Defines augmentation for training data. Each technique applied with a probability.
    
    Parameters:
    -----------
        im_size : int
            size of the image
    
    Return:
    -------
        train_transform : albumentations.compose
    """
    if augment_mode == '1':
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(),  
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=0.8,
            ),   
            A.Rotate(interpolation=0),    
            A.RandomSizedCrop(min_max_height=[int(0.5*im_size), int(0.8*im_size)], height=im_size, width=im_size, interpolation=0, p=0.5),  
        ]

    if augment_mode == '2':
        print('hello')
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(),   
            A.Rotate(interpolation=0),     
        ]
    
    if augment_mode == '3':
        train_transform = [  
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=0.8,
            ),    
        ]

    if augment_mode == '4':
        train_transform = [  
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(),
        ]

    return A.Compose(train_transform)

"""
def get_preprocessing(preprocessing_fn):
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)
"""

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing():
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

def expand_greyscale_channels(image):
    # add channel dimension
    image = np.expand_dims(image, -1)
    # copy last dimension to reach shape of RGB
    image = image.repeat(3, axis=-1)
    return image

def compute_class_weights(masks_dir):
    train_masks = np.load(masks_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    masks_resh = train_masks.reshape(-1,1)
    masks_resh_list = masks_resh.flatten().tolist()
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(masks_resh), y=masks_resh_list)

    print(class_weights.shape)
    return class_weights