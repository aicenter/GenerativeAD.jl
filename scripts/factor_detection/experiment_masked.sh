#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 40G
#SBATCH --time 24:00:00
#SBATCH --job-name factor_detection
#SBATCH --output logs/jupyter-notebook-%J.log

module load Python/3.9.6-GCCcore-11.2.0
module load Julia/1.5.3-linux-x86_64

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"


while read d; do
	julia --project factor_detection_masked.jl sgvaegan100 wildlife_MNIST $d 
done < wmnist_models.txt
