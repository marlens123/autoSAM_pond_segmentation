#python scripts/main_autosam_seg.py --save_dir './AutoSamPonds3' --normalize
#python scripts/infer_ponds.py --pref 003 --weights_path 'experiments/AutoSamPonds3/model.pth' --normalize

#python scripts/main_autosam_seg.py --save_dir './test' --normalize
#python scripts/main_autosam_seg.py --save_dir './test_lr_small' --normalize --lr 0.0001
#python scripts/infer_ponds.py --pref 003 --weights_path 'experiments/AutoSamPonds3/model.pth' --normalize

#python scripts/main_autosam_seg.py --save_dir './test_class_weights' --normalize --use_class_weights

#python scripts/main_autosam_seg.py --save_dir './default' --normalize

#same as default
#python scripts/main_autosam_seg.py --save_dir './batch_1' --normalize --batch_size 1

# bit better as default
#python scripts/main_autosam_seg.py --save_dir './batch_4' --normalize --batch_size 4

# performance drop
#python scripts/main_autosam_seg.py --save_dir './augment_3' --normalize --augmentation

# best results
#python scripts/main_autosam_seg.py --save_dir './augment_2' --normalize --augmentation --augment_mode 2

#python scripts/main_autosam_seg.py --save_dir './augment_2_2' --normalize --augmentation --augment_mode 2 --pref 'aug2_2'

# bad results
#python scripts/main_autosam_seg.py --save_dir './augment_2_3' --normalize --augmentation --batch-size 4 --epochs 200 --augment_mode 2 --pref 'aug2_3'

#python scripts/main_autosam_seg.py --save_dir './dropout' --normalize --augmentation --dropout --epochs 200 --augment_mode 2 --pref 'dropout'
#python scripts/main_autosam_seg.py --save_dir './augment_2_4' --normalize --augmentation --batch-size 1 --augment_mode 2 --pref 'batch1'

#python scripts/main_autosam_seg.py --save_dir './dropout_20' --normalize --augmentation --dropout --epochs 200 --augment_mode 2 --pref 'dropout_20'


#python scripts/main_autosam_seg.py --save_dir './augment_2_5' --normalize --epochs 300 --augmentation --augment_mode 2 --pref 'aug2_5'

#python scripts/main_autosam_seg.py --save_dir './new_data' --normalize --epochs 300 --augmentation --augment_mode 2 --pref 'new_data'

#python scripts/main_autosam_seg.py --save_dir './2201_2' --normalize --epochs 300 --augmentation --augment_mode 2 --pref '2201_2'

#python scripts/main_autosam_seg.py --save_dir './2201_2_no_1' --normalize --epochs 150 --pref '2201_2_no_1'
#python scripts/main_autosam_seg.py --save_dir './2201_2_no_2' --normalize --epochs 150 --pref '2201_2_no_2'
#python scripts/main_autosam_seg.py --save_dir './2201_2_no_3' --normalize --epochs 150 --pref '2201_2_no_3'
#python scripts/main_autosam_seg.py --save_dir './2201_2_no_4' --normalize --epochs 150 --pref '2201_2_no_4'

python scripts/main_autosam_seg.py --save_dir './2301_1' --normalize --epochs 150 --augmentation --augment_mode 2 --pref '2301_1'
python scripts/main_autosam_seg.py --save_dir './2301_2' --normalize --epochs 150 --augmentation --augment_mode 2 --pref '2301_2'
python scripts/main_autosam_seg.py --save_dir './2301_3' --normalize --epochs 150 --augmentation --augment_mode 2 --pref '2301_3'
python scripts/main_autosam_seg.py --save_dir './2301_4' --normalize --epochs 150 --augmentation --augment_mode 2 --pref '2301_4'