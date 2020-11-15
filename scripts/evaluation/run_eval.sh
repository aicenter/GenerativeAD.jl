#!/bin/bash
#SBATCH --partition cpufast
#SBATCH --nodes 1
#SBATCH --cpus-per-task 32
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 4:00:00
#SBATCH --job-name generativead-eval
#SBATCH --qos==collaborator

DATASETS=$1
SUFFIX=$2
if [ "$DATASETS" == "tabular" ] || [ "$DATASETS" == "images" ]; then
    module load Julia/1.5.1-linux-x86_64
    module load Python/3.8.2-GCCcore-9.3.0

    julia --threads 32 --project ./generate_stats.jl experiments${SUFFIX}/${DATASETS} evaluation${SUFFIX}/${DATASETS} 
    julia --threads 32 --project ./collect_stats.jl  evaluation${SUFFIX}/${DATASETS} evaluation${SUFFIX}/${DATASETS}_eval.bson -f
else
    echo "Unsupported dataset type. Only 'images' or 'tabular' are supported."
fi