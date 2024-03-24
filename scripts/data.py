# inspired by https://github.com/qubvel/segmentation_models.pytorch

import numpy as np

from .utils import (
    get_training_augmentation,
    get_preprocessing,
    expand_greyscale_channels,
)

from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        mode (str): Image mode ('train' or 'test')
    """

    CLASSES = ["melt_pond", "sea_ice", "ocean"]

    def __init__(
        self,
        args,
        mode,
        preprocessing=get_preprocessing(),
        classes=["melt_pond", "sea_ice"],
    ):
        self.mode = mode

        if self.mode == "train":
            self.images_fps = np.load(args.images_train_dir).tolist()
            self.masks_fps = np.load(args.masks_train_dir).tolist()
        elif self.mode == "test":
            self.images_fps = np.load(args.images_test_dir).tolist()
            self.masks_fps = np.load(args.masks_test_dir).tolist()
        else:
            print("Specified mode must be either 'train' or 'test'")

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.normalize = args.normalize

        self.augmentation = args.augmentation
        self.augment_mode = args.augment_mode

        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = self.images_fps[i]
        # reshape to 3 dims in last channel
        image = expand_greyscale_channels(image)

        mask = self.masks_fps[i]
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=-1)

        print(mask.shape)
        print(np.unique(mask))

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # apply normalization
        if self.normalize:
            # image = (image - image.min()) / (image.max() - image.min())
            image = (image - image.mean()) / image.std()

        if self.mode == "train" and self.augmentation:
            augmentation = get_training_augmentation(
                im_size=480, augment_mode=self.augment_mode
            )
            sample = augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.images_fps)
