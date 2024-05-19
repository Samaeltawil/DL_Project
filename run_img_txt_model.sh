#!/bin/bash -l
#SBATCH --chdir /home/coderey/project/DL_Project
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 03:00:00
#SBATCH --gres gpu:1
#SBATCH --account ee-559
#SBATCH --qos ee-559

# The --reservation line only works during the class.
# conda activate hatespeech
echo $CONDA_PREFIX
echo "$PWD"
python run_img_txt_model.py 