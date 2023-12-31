# AutoSAM for Melt Pond Detection 
(work in progress)

This repository adapts AutoSAM introduced by Xinrong Hu et al. ([Link]() to paper) to the segmentation of Arctic thermal infrared images.

[Link](https://github.com/marlens123/pond_segmentation) to the U-net, Att-U-net and PSP-net implementation of the same task.


## Setup
This code requires `python>=3.10`, as well as `pytorch>=1.7` and `torchvision>=0.8`.  Install additional packages using ```pip install -r requirements.txt``.

SAM model checkpoints can be downloaded from [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) and should be placed in 'segment_anything_checkpoints'.

The TIR dataset is planned to be published in the future.

## How to use
### Predict flight images with fine-tuned AutoSAM
```
python scripts/main_feat_seg.py --src_dir ${ACDC_folder} \
--data_dir ${ACDC_folder}/imgs/ --save_dir ./${output_dir}  \
--b 4 --dataset ACDC --gpu ${gpu} \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4
```
${tr_size} decides how many volumes used in the training; ${model_type} is selected from vit_b (default), vit_l, and vit_h;

### Finetune AutoSAM
Requires GPU (change gpu name with --gpu ${GPU_name})

```
python scripts/main_autosam_seg.py --save_dir ${storage_folder_name} --normalize
```

## Credits
This project is based on the work of [Original Author's Name](link-to-original-repo), licensed under the Apache 2.0 License.

Modifications:
- Explain the main changes or adaptations you made to the original code.
- Add any contributors or team members who have significantly contributed to the modified codebase.

[[`Paper`](https://arxiv.org/pdf/2306.13731.pdf)]

![](./autosam.png)


## Citation
If you find our codes useful, please cite
```
@article{hu2023efficiently,
  title={How to Efficiently Adapt Large Segmentation Model (SAM) to Medical Images},
  author={Hu, Xinrong and Xu, Xiaowei and Shi, Yiyu},
  journal={arXiv preprint arXiv:2306.13731},
  year={2023}
}
```
