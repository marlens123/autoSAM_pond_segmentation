# AutoSAM for Melt Pond Detection 

This repository adapts AutoSAM to the segmentation of helicopter-borne Arctic thermal infrared images. The segmentation should be done into three classes: melt ponds, sea ice, and ocean.

Training and validation data are stored in ```data/training/``` and we provide the weights of our final model in ```experiments/2801_2/``` (lfs tracked).
Test images are stored in ```data/prediction/preprocessed/test/```.

![results_2701_test](https://github.com/marlens123/autoSAM_pond_segmentation/assets/80780236/e5ed5387-fe31-449a-b9dc-f2625a3bae20)


## Setup
This code requires `python>=3.10`, as well as `pytorch>=1.7` and `torchvision>=0.8`.  Install additional packages using ```pip install -r requirements.txt```.

Segment Anything model checkpoints can be downloaded from [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) and should be placed in ```segment_anything_checkpoints/```.

The entire TIR dataset will be published in the future.

## How to use
GPU required.

### Predict validation images with fine-tuned AutoSAM
```
python -m scripts.infer_ponds --pref ${storage_folder_name} --weights_path 'experiments/2801_2/model_mp_145.pth' --skip_preprocessing --preprocessed_path 'data/prediction/preprocessed/val/' --val_predict
```
Images are then stored here ```data/prediction/predicted/${storage_folder_name}```.

The computed melt pond fraction will be stored in ```metrics/melt_pond_fraction/${storage_folder_name}```.

### Predict test images with fine-tuned AutoSAM
```
python -m scripts.infer_ponds --pref ${storage_folder_name} --weights_path 'experiments/2801_2/model_mp_145.pth' --skip_preprocessing --preprocessed_path 'data/prediction/preprocessed/test/' --val_predict
```
Images are then stored here ```data/prediction/predicted/${storage_folder_name}```.

The computed melt pond fraction will be stored in ```metrics/melt_pond_fraction/${storage_folder_name}```.

### Finetune AutoSAM
```
python -m scripts.main_autosam_seg --save_dir ${storage_folder_name} --pref ${wandb_pref_name_of_choice}
```

```preprocess_training.ipynb``` was used to preprocess the data.

## Credits
The AutoSAM implementation is based on the work of [Hu, Xinrong and Xu, Xiaowei and Shi, Yiyu](https://github.com/xhu248/AutoSAM), licensed under the Apache 2.0 License. Modifications are listed in the respective files.

Full citation of AutoSAM:
```
@article{hu2023efficiently,
  title={How to Efficiently Adapt Large Segmentation Model (SAM) to Medical Images},
  author={Hu, Xinrong and Xu, Xiaowei and Shi, Yiyu},
  journal={arXiv preprint arXiv:2306.13731},
  year={2023}
}
```

## Further Reference
[1] Kanzow, Thorsten (2023). The Expedition PS131 of the Research Vessel POLARSTERN to the
Fram Strait in 2022. Ed. by Horst Bornemann and Susan Amir Sawadkuhi. Bremerhaven. DOI: 10.57738/BzPM\_0770\_2023.

Contact: marlena1@gmx.de
