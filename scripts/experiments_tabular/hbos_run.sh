#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=20G

MAX_SEED=$1
DATASET=$2
HP_SAMPLING=$3
CONTAMINATION=$4

module load Julia/1.5.1-linux-x86_64
module load Python/3.8.2-GCCcore-9.3.0

# load virtualenv
source ${HOME}/sklearn-env/bin/activate
export PYTHON="${HOME}/sklearn-env/bin/python"

# PyCall needs to be rebuilt if environment changed
julia --project -e 'using Pkg; Pkg.build("PyCall"); @info("SETUP DONE")'

julia ./hbos.jl ${MAX_SEED} $DATASET ${HP_SAMPLING} $CONTAMINATION
