import numpy as np
import cv2
import argparse
from scripts.utils import transform_color

parser = argparse.ArgumentParser()

parser.add_argument(
    "--date", default="220719_2", type=str, help="Date of the flight to be added."
)
parser.add_argument("--flight_nr", default="flight_11", type=str, help="Flight number.")

parser.add_argument(
    "--path_to_X_train",
    default="data/training/flight9_16_7_11/train_images.npy",
    type=str,
    help="Path to current training images in .npy file format.",
)
parser.add_argument(
    "--path_to_y_train",
    default="data/training/flight9_16_7_11/train_masks.npy",
    type=str,
    help="Path to current training masks in .npy file format.",
)


def main():
    args = parser.parse_args()
    params = vars(args)

    # indeces = [8]
    indeces = [8, 40]

    # masks = ['f7_8_mask']
    masks = ["f11_8_mask", "f11_40_mask"]

    # path = 'data/prediction/predicted/{}/raw/'.format(params['flight_nr'])

    masks_to_add = []
    imgs_to_add = []

    for i, idx in enumerate(indeces):
        mask = cv2.imread("{}.png".format(masks[i]), 0)
        mask = transform_color(mask)
        mask[mask == 20] = 2
        image = cv2.imread(
            "data/prediction/preprocessed/{0}/{1}.png".format(params["date"], idx), 0
        )

        imgs_to_add.append(image)
        masks_to_add.append(mask)

    masks_to_add = np.array(masks_to_add)
    imgs_to_add = np.array(imgs_to_add)

    images = np.load(params["path_to_X_train"])
    masks = np.load(params["path_to_y_train"])

    new_images = np.concatenate((images, imgs_to_add), axis=0)
    new_masks = np.concatenate((masks, masks_to_add), axis=0)

    print("New shape images: {}".format(new_images.shape))
    print("New shape masks: {}".format(new_masks.shape))

    np.save(params["path_to_X_train"], new_images)
    np.save(params["path_to_y_train"], new_masks)


if __name__ == "__main__":
    main()
