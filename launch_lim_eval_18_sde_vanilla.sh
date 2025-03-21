#!/bin/bash

main_dir=/home/ma-user/work/lim-exps/code/LIM-final-code_v0.1
config=$main_dir/configs/cifar10_ncsnpp_deep.yml
dataset='/home/ma-user/work/lim-exps/data'

exp='/home/ma-user/work/lim-exps/output/exps/lim-ncsnpp_deep'
exp_name='cifar10_18'

fid=test
if [ $fid == "test" ]
then
    n_samples='10k_samples'
else 
    n_samples='50k_samples'
fi

ckpt='100'
ckpt_dir=/home/ma-user/work/lim-exps/output/exps/lim-ncsnpp_deep/logs/$exp_name/ckpt_"$ckpt"000.pth

sample_type='sde_vanilla'
solver_type='exponential_integrator'
nfe=20

img_folder=/home/ma-user/work/lim-exps/output/exps/lim-ncsnpp_deep/images/$exp_name/"$ckpt"k_ckpt/"$n_samples"_"$sample_type"_"$solver_type"_"$nfe"
cd $main_dir

echo $sample_type

export TF_ENABLE_ONEDNN_OPTS=0 && torchrun --rdzv_backend c10d \
                                           --rdzv_endpoint localhost:29491 \
                                            main.py \
                                            --config $config --seed 42 --dataset $dataset \
                                            --exp $exp --sample --ddp --fid $fid\
                                            --sample_type $sample_type --solver_type $solver_type \
                                            --image_folder $img_folder --ckpt_dir $ckpt_dir \
                                            --nfe $nfe --alpha 1.8
