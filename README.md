# AutoSAM for Melt Pond Detection 

This repository adapts AutoSAM (credits in the paper) to the segmentation of helicopter-borne Arctic thermal infrared images. The segmentation should be done into three classes: melt ponds, sea ice, and ocean.

Training and validation data are stored in ```data/training/``` and we provide the weights of our final model in ```experiments/2801_2/``` (lfs tracked).
Test images are stored in ```data/prediction/preprocessed/test/```.

## Setup
This code requires `python>=3.10`, as well as `pytorch>=1.7` and `torchvision>=0.8`.  Install additional packages using ```pip install -r requirements.txt```.

Segment Anything model checkpoints can be downloaded from [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) and should be placed in ```segment_anything_checkpoints/```.

The entire TIR dataset will be published in the future.

## How to use
GPU required.

### Predict images with fine-tuned AutoSAM

```
python scripts/infer_ponds.py --pref ${storage_folder_name} --data 'data/prediction/raw/${path_to_nc_file_to_be_predicted}' --weights_path 'experiments/2801_2/model_mp_145.pth' --normalize
```
(full data for prediction currently not provided)

### Finetune AutoSAM
```
python scripts/main_autosam_seg.py --save_dir ${storage_folder_name} --normalize
python scripts/main_autosam_seg.py --save_dir ${storage_folder_name} --normalize --epochs 150 --augmentation --augment_mode 2 --pref ${pref_name_of_choice} --use_class_weights
```

```preprocess_training.ipynb``` was used to preprocess the data.

## Credits
The AutoSAM implementation is based on the work of the original AutoSAM authors (which are credited in the paper), licensed under the Apache 2.0 License. Modifications are listed in the respective files.
