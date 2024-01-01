# AutoSAM for Melt Pond Detection 
(work in progress)

This repository adapts AutoSAM introduced by Xinrong Hu et al. ([Link](https://arxiv.org/pdf/2306.13731.pdf) to author's paper) to the segmentation of Arctic thermal infrared images.

![compare](https://github.com/marlens123/autoSAM_pond_segmentation/assets/80780236/796e8e4f-0e8c-41bd-b614-e4273b0cbc17)
*(code for Att-Unet implementation can be found [here](https://github.com/marlens123/pond_segmentation)*)

## Setup
This code requires `python>=3.10`, as well as `pytorch>=1.7` and `torchvision>=0.8`.  Install additional packages using ```pip install -r requirements.txt```.

SAM model checkpoints can be downloaded from [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) and should be placed in 'segment_anything_checkpoints/'.

The TIR dataset will be published in the future. The training data used in this work can be accessed from [here](https://drive.google.com/drive/folders/1IWzR09t3Visb1Jy8a8rsvbERpgZpYaB0?usp=drive_link). Training data should be placed in 'dataset/melt_ponds/'.

The weights of the current model configuration can be downloaded from [here](https://drive.google.com/drive/folders/1Dm9pOtBx5CKlAI21p_ACf-pwEICZRzYP?usp=drive_link) and should be placed in 'experiments/AutoSamPonds3/'.

## How to use
GPU required.

### Predict images with fine-tuned AutoSAM

(data currently not provided)
```
python scripts/infer_ponds.py --pref ${storage_folder_name} --data 'dataset/melt_pond_prediction/raw/${path_to_nc_file_to_be_predicted}' --weights_path 'experiments/AutoSamPonds3/model.pth' --normalize
```

### Finetune AutoSAM
```
python scripts/main_autosam_seg.py --save_dir ${storage_folder_name} --normalize
```

## TO-DOs
- develop more reliable evaluation method (the small amount of labeled data makes it difficult to provide a reliable metric; current evaluation is mainly based on qualitative inspection; k-crossfold validation implementation work in progress)
- label more data for difficult cases
- hyperparameter optimization
- test vit_l and vit_h (memory of currently used GPU too small)

## Credits
This project is based on the work of [Hu, Xinrong and Xu, Xiaowei and Shi, Yiyu](https://github.com/xhu248/AutoSAM), licensed under the Apache 2.0 License.

[Corresponding [`Paper`](https://arxiv.org/pdf/2306.13731.pdf)]:
```
@article{hu2023efficiently,
  title={How to Efficiently Adapt Large Segmentation Model (SAM) to Medical Images},
  author={Hu, Xinrong and Xu, Xiaowei and Shi, Yiyu},
  journal={arXiv preprint arXiv:2306.13731},
  year={2023}
}
```
