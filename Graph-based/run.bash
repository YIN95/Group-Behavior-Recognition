#!/usr/bin/env bash
#SBATCH --mem  300GB
#SBATCH --gres gpu:8
#SBATCH --cpus-per-task 8
#SBATCH --constrain "khazadum|rivendell|belegost|"
#SBATCH --mail-type FAIL
#SBATCH --output /Midgard/home/wyin/repo/out_train.log
#SBATCH --error /Midgard/home/wyin/repo/error_train.log

printenv $SLURM_STEP_GPUS
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate tr
python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train.yaml