import sys
import os

# add parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import re
import cv2
import csv
import netCDF4
import argparse
import matplotlib.pyplot as plt
from scripts.utils import calculate_mpf, predict_image, crop_center_square, label_to_pixelvalue

parser = argparse.ArgumentParser(description="Uses trained model to predict and store surface masks from netCDF file containing TIR images from a single helicopter flight. Optional calculation of melt pond fraction (MPF).")

parser.add_argument("--pref", type=str, default="001", help="Identifier for the current prediction. Will be used as foldername to store results.")

parser.add_argument("--data", default="IRdata_ATWAICE_preprocessed_220718_142920.nc", type=str, help="Either: 1) Filename of netCDF data file. For this, data must be stored in 'data/prediction/raw'. Or: 2) Absolute path to netCDF data file. Then data must not be copied in advance.")
parser.add_argument("--weights_path", default="experiments/AutoSamPonds3/model.pth", type=str, help="Path to model weights that should be used. Must contain the model architecture as second-to-last part of path (should be per default).")
parser.add_argument("--preprocessed_path", default="data/prediction/preprocessed", type=str, help="Path to folder that should store the preprocessed images.")
parser.add_argument("--predicted_path", default="data/prediction/predicted", type=str, help="Path to folder that should store the predicted image masks.")
parser.add_argument("--metrics_path", default="metrics/melt_pond_fraction/", type=str, help="Path to .csv file that should store the resulting mpf (if calculation is desired).")
parser.add_argument("--model_type", default="vit_b", type=str, help="Model type that should be used. Must be the same as in 'weights_path'.")
parser.add_argument("--skip_mpf", default=False, action="store_true", help="Skips the calculation of the melt pond fraction for the predicted flight.")
parser.add_argument("--skip_preprocessing", default=False, action="store_true", help="Skips preprocessing. Can be used to directly perform mpf calculation. In that case, 'predicted_path' must contain predicted images.")
parser.add_argument("--skip_prediction", default=False, action="store_true", help="Skips prediction process. Can be used to directly perform mpf calculation. In that case, 'predicted_path' must contain predicted images.")
parser.add_argument("--convert_to_grayscale", default=True, action="store_false", help="Converts predicted images to grayscale for visualization and stores in 'data/prediction/predicted/[pref]/grayscale'.")
parser.add_argument("--normalize", default=False, action="store_true", help="Normalize images before prediction. Should be used if model was trained with normalized images.")
parser.add_argument("--no_finetune", default=False, action="store_true", help="Skips finetuning of model. Should be used if model was trained without finetuning.")
parser.add_argument("--model", type=str, default="autosam")

def main():
    args = parser.parse_args()
    params = vars(args)

    # add prefix to storage paths and create folder
    params['predicted_path'] = os.path.join(params['predicted_path'], params['pref'])
    os.makedirs(params['predicted_path'], exist_ok = True)
    params['metrics_path'] = os.path.join(params['metrics_path'], params['pref'])
    os.makedirs(params['metrics_path'], exist_ok = True)

    if params['data'] == "none":
        print("Data is none. Must be specified.")

    id = params['data'].split('/')[-2]
    print(id)

    # extract date of flight used
    match = re.search(r"(\d{6})_(\d{6})", params['data'])

    if match:
        date_part = match.group(1)

        # formatting the date
        formatted_date = f"20{date_part[:2]}-{date_part[2:4]}-{date_part[4:]}"
        print(f"The date in the filename is: {formatted_date}")

        params['preprocessed_path'] = os.path.join(params['preprocessed_path'], id)
        os.makedirs(params['preprocessed_path'], exist_ok = True)

    else:
        print("Date not found in the filename.")

    # extract model architecture from weights_path
    model_arch = params['weights_path'].split('/')[1]
    print(model_arch)
    print(type(model_arch))
    print("Model architecture used: ".format(model_arch))

    if not params['skip_preprocessing']:

        # load data and store as images
        # use whole path when abs path is given, else use data from 'data/prediction/raw'
        if '/' in params['data']:
            ds = netCDF4.Dataset(params['data'])
            print("Abs path is used.")
        else:
            ds = netCDF4.Dataset(os.path.join('data/prediction/raw', id, params['data']))
            print("Rel path is used.")
        imgs = ds.variables['Ts'][:]

        tmp = []

        for im in imgs:
            im = crop_center_square(im)
            tmp.append(im)

        imgs = tmp

        print("Start extracting images...")

        # extract only every 4th image to avoid overlap
        for idx, img in enumerate(imgs):
            if(idx % 4 == 0):
                plt.imsave(os.path.join(params['preprocessed_path'], '{}.png'.format(idx)), img, cmap='gray')

    if not params['skip_prediction']:

        print("Start predicting images...")

        # extract surface masks from images
        for idx, file in enumerate(os.listdir(params['preprocessed_path'])):
            os.makedirs(os.path.join(params['predicted_path'], 'raw/'), exist_ok = True)
            id = file.split('.')[0]

            if file.endswith('.png'):
                img = cv2.imread(os.path.join(params['preprocessed_path'], file), 0)
                predict_image(img, 480, params['weights_path'], model_type=params['model_type'], backbone='resnet34', train_transfer='imagenet', save_path=os.path.join(params['predicted_path'],'raw/{}.png'.format(id)), visualize=False, normalize=params['normalize'], no_finetune=params['no_finetune'], model_arch=params['model'])

    # optionally convert to grayscale images for visibility
    if params['convert_to_grayscale']:
        os.makedirs(os.path.join(params['predicted_path'], 'grayscale/'), exist_ok = True)

        for idx, file in enumerate(os.listdir(os.path.join(params['predicted_path'],'raw/'))):
            id = file.split('.')[0]
            im = label_to_pixelvalue(cv2.imread(os.path.join(params['predicted_path'],'raw/', file)))
            cv2.imwrite(os.path.join(params['predicted_path'],'grayscale/{}.png'.format(id)), im)


    # optionally calculate melt pond fraction and store in csv file
    if not params['skip_mpf']:
        mpf = calculate_mpf(os.path.join(params['predicted_path'], 'raw/'))

        headers = ['flight_date', 'melt_pond_fraction']

        with open(os.path.join(params['metrics_path'], 'mpf.csv'), 'a', newline='') as f:
            writer = csv.writer(f)

            # headers in the first row
            if f.tell() == 0:
                writer.writerow(headers)

            writer.writerow([formatted_date, mpf])

    print("Process ended.")

if __name__ == "__main__":
    main()

    



