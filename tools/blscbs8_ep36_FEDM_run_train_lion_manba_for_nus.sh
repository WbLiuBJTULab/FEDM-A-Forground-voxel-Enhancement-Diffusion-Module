#! /bin/bash


## kitti mamba
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/FEDM_lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml \
--extra_tag ep36_FEDM_lion_mamba_nusc_8x_1f_1x_one_stride_128dim \
--batch_size 8  --epochs 36 --max_ckpt_save_num 36 --workers 4 --sync_bn



