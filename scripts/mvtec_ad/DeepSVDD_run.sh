#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=80G

MAX_SEED=$1
CATEGORY=$2
CONTAMINATION=$3

module load Julia/1.5.1-linux-x86_64
module load Python/3.8.2-GCCcore-9.3.0

# PyCall needs to be rebuilt if environment changed
julia --project -e 'using Pkg; Pkg.build("PyCall"); @info("SETUP DONE")'

julia --project -e 'using Pkg; Pkg.instantiate();'

julia ./DeepSVDD.jl ${MAX_SEED} $CATEGORY $CONTAMINATION
