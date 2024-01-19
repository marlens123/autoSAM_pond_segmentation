#python scripts/main_autosam_seg.py --save_dir './AutoSamPonds3' --normalize
#python scripts/infer_ponds.py --pref 003 --weights_path 'experiments/AutoSamPonds3/model.pth' --normalize

#python scripts/main_autosam_seg.py --save_dir './test' --normalize
#python scripts/main_autosam_seg.py --save_dir './test_lr_small' --normalize --lr 0.0001
#python scripts/infer_ponds.py --pref 003 --weights_path 'experiments/AutoSamPonds3/model.pth' --normalize

#python scripts/main_autosam_seg.py --save_dir './test_class_weights' --normalize --use_class_weights

python scripts/main_autosam_seg.py --save_dir './default' --normalize

python scripts/main_autosam_seg.py --save_dir './batch_1' --normalize --batch_size 1

python scripts/main_autosam_seg.py --save_dir './batch_4' --normalize --batch_size 4

python scripts/main_autosam_seg.py --save_dir './augment_3' --normalize --augmentation

python scripts/main_autosam_seg.py --save_dir './augment_2' --normalize --augmentation --augment_mode 2