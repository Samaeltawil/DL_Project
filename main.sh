#!/bin/bash -l
#SBATCH --chdir /home/adpannat/DL_Project
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 08:00:00
#SBATCH --gres gpu:1
#SBATCH --account ee-559
#SBATCH --qos ee-559

# The --reservation line only works during the class.
conda activate DL_env
echo $CONDA_PREFIX
echo "$PWD"
python main.py 