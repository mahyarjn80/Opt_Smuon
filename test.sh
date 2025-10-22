#!/bin/bash
#SBATCH -J myjob_4GPUs
#SBATCH -o ./logs/myjob_4GPUs_%j.out
#SBATCH -e ./logs/myjob_4GPUs_%j.err
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
cd $HOME/centralflows

module load miniforge

export PYTHONUNBUFFERED=TRUE

python main_dual_training.py \
    arch:cifar-net \
    --optimizer-configs-str \
    	"Shampoo:lr=0.24,momentum=0.9,order_multiplier=1" \
    	"Shampoo:lr=0.24,momentum=0.9,order_multiplier=2" \
    	"Muon:lr=0.24,momentum=0.9" \
    --batch-sweep-count 20 --batch-size 16192
