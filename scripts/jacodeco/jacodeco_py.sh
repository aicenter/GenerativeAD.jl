#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 60G
#SBATCH --time 24:00:00
#SBATCH --job-name jacodeco
#SBATCH --output $HOME/logs/jacodeco/%J.log

MODEL=$1
DATASET=$2
AC=$3
FORCE=$4

module load Julia/1.5.3-linux-x86_64
module load Python/3.9.6-GCCcore-11.2.0

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"
julia --project -e 'using Pkg; Pkg.build("PyCall"); @info("SETUP DONE")'

julia ./jacodeco_py.jl $MODEL $DATASET $AC $FORCE

