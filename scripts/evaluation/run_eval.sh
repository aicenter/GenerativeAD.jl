#!/bin/bash
#SBATCH --partition cpufast
#SBATCH --nodes 1
#SBATCH --cpus-per-task 32
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 4:00:00
#SBATCH --job-name generativead-eval
#SBATCH --output eval-%J.out
#SBATCH --qos==collaborator

module load Julia/1.5.1-linux-x86_64
module load Python/3.8.2-GCCcore-9.3.0

# load virtualenv
source ${HOME}/julia-env/bin/activate
export PYTHON="${HOME}/julia-env/bin/python"

julia --threads 32 --project ./generate_stats.jl experiments/images evaluation/images 
julia --threads 32 --project ./collect_stats.jl  evaluation/images evaluation/images_eval.bson -f
