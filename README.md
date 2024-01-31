# AutoSAM for Melt Pond Detection 

(Training and test data is stored in ```data/training/``` and we provide the weights of our final model in ```experiments/2801_2/``` (lfs tracked)).

This repository adapts AutoSAM introduced by Xinrong Hu et al. ([Link](https://arxiv.org/pdf/2306.13731.pdf) to authors' paper) to the segmentation of helicopter-borne Arctic thermal infrared images. The segmentation should be done into three classes: melt ponds, sea ice, and ocean. A particular challenge for TIR imagery is the presence of spatially and temporally varying surface temperatures, so that classification cannot be based on spectral features alone.

## Current Results
(more examples are uploaded to ```predictions/```, where left=input, middle=Att-Unet, right=AutoSAM)

![compare_240103_rand](https://github.com/marlens123/autoSAM_pond_segmentation/assets/80780236/49797ca3-7c5d-414f-874c-803835b342ba)
*(code for Att-Unet and PSP-Net implementation can be found [here](https://github.com/marlens123/pond_segmentation)*)

## Setup
This code requires `python>=3.10`, as well as `pytorch>=1.7` and `torchvision>=0.8`.  Install additional packages using ```pip install -r requirements.txt```.

SAM model checkpoints can be downloaded from [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) and should be placed in ```segment_anything_checkpoints/```.

The TIR images used were acquired during the PS131 ATWAICE Campaign [1]. The entire dataset will be published in the future.

## How to use
GPU required.

### Predict images with fine-tuned AutoSAM

```
python scripts/infer_ponds.py --pref ${storage_folder_name} --data 'data/prediction/raw/${path_to_nc_file_to_be_predicted}' --weights_path 'experiments/AutoSamPonds3/model.pth' --normalize
```
(full data for prediction currently not provided)

### Finetune AutoSAM
```
python scripts/main_autosam_seg.py --save_dir ${storage_folder_name} --normalize
```

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

This repo is based on my work as a student assistant in the [Remote Sensing of Polar Regions group](https://seaice.uni-bremen.de/research-group/), University of Bremen under the supervision of Dr. Gunnar Spreen.

**Contact**: mareil@uni-osnabrueck.de
