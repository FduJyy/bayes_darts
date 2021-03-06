#!/usr/bin/env bash
# nvidia-smi
# mount -t tmpfs -o size=1G tmpfs /userhome/temp_data
# cp -r /userhome/data/cifar10 /userhome/temp_data/
# pip3 install -U torch==1.2.0
# pip3 install -U torchvision==0.2.2.post3
python augment_hsd.py --name mde_1 --dataset iccad19 --gpus all --batch_size 128 --workers 16 --lr 0.1 --print_freq 50 --fealen 32 --genotype "Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
        [('skip_connect', 0), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 3), ('sep_conv_3x3', 1)],
        [('dil_conv_5x5', 3), ('max_pool_3x3', 4)],
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
        [('skip_connect', 0), ('skip_connect', 1)],
        [('sep_conv_3x3', 3), ('skip_connect', 2)],
        [('dil_conv_3x3', 3), ('sep_conv_5x5', 0)],
    ],
    reduce_concat=range(2, 6))"