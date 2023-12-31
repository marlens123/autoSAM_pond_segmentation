import sys
import os

# add parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from dataset.melt_ponds.scripts.utils import get_preprocessing

from models.build_autosam_seg_model import sam_seg_model_registry as autosam
from models.build_sam_feat_seg_model import sam_feat_seg_model_registry as featseg

def visualize_ir(img, idx=None, cmap='cividis', colorbar=False, save_path=None):
    """
    For visualization of ir images.
    """
    plt.imshow(img, cmap=cmap)

    if colorbar:
        plt.colorbar()
    
    if not save_path==None:
        #cv2.imwrite(os.path.join(save_path, '{}.png'.format(idx)), img)
        plt.imsave(os.path.join(save_path, '{}.png'.format(idx)), img, cmap='gray')

def expand_greyscale_channels(image):
    """
    Copies last channel three times to reach RGB-like shape.
    """
    image = np.expand_dims(image, -1)
    image = image.repeat(3, axis=-1)
    return image


def crop_center_square(image, im_size=480):
    """"
    Crops the center of the input image with specified size.
    """
    size=im_size
    height, width = image.shape[:2]
    new_width = new_height = size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    cropped_image = image[top:bottom, left:right]
    return cropped_image

def label_to_pixelvalue(image):
    """
    Transforms class labels to pixelvalues in the grayscale range to be able to make outcomes visible.
    """
    uniques = np.unique(image)
    
    for idx,elem in enumerate(uniques):
        mask = np.where(image == 1)
        image[mask] = 125
        mask2 = np.where(image == 2)
        image[mask2] = 255
    return image

def preprocess_prediction(image, model_preprocessing, smooth=False, normalize=False):
    """
    Preprocesses image to be suitable as input for model prediction.
    """
    image = expand_greyscale_channels(image)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create mask of zeros such that preprocessing function works
    random_mask = np.zeros((image.shape[0], image.shape[1], 1))

    image = image.astype(np.float32)

    # apply normalization
    if normalize:
        image = (image - image.min()) / (image.max() - image.min())

    sample = model_preprocessing(image=image, mask=random_mask)
    image, _ = sample['image'], sample['mask']

    if not smooth:
        # will add a dimension that replaces batch_size
        image = np.expand_dims(image, axis=0)
        # if smooth, function takes care of this

    image = torch.tensor(image, device=device)
    
    return image


def patch_predict(model, image, patch_size, model_preprocessing, visualize=True):
    """
    Predicts on image patches and recombines masks to whole image later.
    
    This function is inspired by
    https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_custom_patch_inference.py
    """

    # initialize mask with zeros
    segm_img = np.zeros(image.shape[:2])
    patch_num=1
    # Iterates through image in steps of patch_size, operates on patches
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            single_patch = image[i:i+patch_size, j:j+patch_size]
            single_patch_shape = single_patch.shape[:2]
            single_patch = preprocess_prediction(single_patch, model_preprocessing=model_preprocessing)
            pr_mask = model.predict(single_patch)
            # removes batch dimension and channel dimension by replacing the latter with class with maximum probability value
            fin = np.argmax(pr_mask.squeeze(), axis=2)

            if visualize:
                fin = label_to_pixelvalue(fin)
            # recombine to complete image
            segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(fin, single_patch_shape[::-1])
            print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1

    return segm_img

def single_patch_predict(model, image, model_preprocessing, visualize=True, normalize=False, model_arch='autosam'):

    image = preprocess_prediction(image, model_preprocessing=model_preprocessing, normalize=normalize)
    print("Image shape: " + str(image.shape))

    if model_arch == 'autosam':
        mask, _ = model.forward(image)

        print("Mask shape: " + str(mask.shape))
        mask = np.array(mask.cpu().detach())
        print(np.unique(mask))
        fin = np.argmax(mask.squeeze(axis=1), axis=0)
        print("Mask shape after squeeze: " + str(fin.shape))
        print(np.unique(fin))

    elif model_arch == 'featseg':
        mask = model.forward(image)

        print("Mask shape: " + str(mask.shape))
        mask = np.array(mask.cpu().detach())
        print(np.unique(mask))
        fin = mask.squeeze(axis=0)
        #fin = np.argmax(mask.squeeze(axis=0), axis=1)
        print("Mask shape after squeeze: " + str(fin.shape))
        print(np.unique(fin))
        #fin = np.argmax(fin, axis=0)
        #print("Mask shape after argmax: " + str(fin.shape))
        #print(np.unique(fin))
        fin = fin.reshape(fin.shape[1], fin.shape[2], fin.shape[0])

    if visualize:
        fin = label_to_pixelvalue(fin)

    return fin


def predict_image(img, im_size, weights, model_type='vit_b', backbone='resnet34', train_transfer='imagenet', smooth=False, save_path=None, visualize=True, normalize=False, no_finetune=False, model_arch='autosam'):
    """
    Preprocesses image for prediction, loads model with weights and uses model to predict segmentation mask.
    """
    BACKBONE = backbone
    TRAIN_TRANSFER = train_transfer
    WEIGHTS = weights

    prepro = get_preprocessing()

    # load model
    if model_type=='vit_h':
        print('hello predict')
        model_checkpoint = 'segment_anything_checkpoints/sam_vit_h_4b8939.pth'
    elif model_type == 'vit_l':
        model_checkpoint = 'segment_anything_checkpoints/sam_vit_l_0b3195.pth'
    elif model_type == 'vit_b':
        model_checkpoint = 'segment_anything_checkpoints/sam_vit_b_01ec64.pth'

    if model_arch == 'autosam':
        model = autosam[model_type](num_classes=3, checkpoint=model_checkpoint)
    if model_arch == 'featseg':
        model = featseg[model_type](num_classes=3, checkpoint=model_checkpoint)

    model = model.cuda(0)
    #predictor = SamPredictor(model)

    # load model weights
    if not no_finetune:
        model.load_state_dict(torch.load(weights))

    if smooth:
        return
        #segmented_image = smooth_patch_predict(model, img, im_size, model_preprocessing=prepro, smooth=True, visualize=visualize)
    else:
        # crop the image to be predicted to a size that is divisible by the patch size used
        if im_size==256:
            img = crop_center_square(img, 256)
        if im_size==128:
            img = crop_center_square(img, 384)
        if im_size==64:
            img = crop_center_square(img, 448)
        segmented_image = single_patch_predict(model, img, model_preprocessing=prepro, visualize=visualize, normalize=normalize, model_arch=model_arch)

    visualize_ir(segmented_image)
    cv2.imwrite(save_path, segmented_image)


def calculate_mpf(dir):

    num_imgs = 0
    mpf_coll = 0

    for f in os.listdir(dir):
        if f.endswith('.png'):
            num_imgs += 1
            im = cv2.imread(os.path.join(dir, f),0)
            pond = np.sum(im==0)
            sea_ice = np.sum(im==1)
            mpf = pond / ( sea_ice + pond )
            mpf_coll += mpf

    mpf = mpf_coll / num_imgs

    return mpf