#!/bin/bash
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 48
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 16G
#SBATCH --time 24:00:00
#SBATCH --job-name generativead-eval
#SBATCH --qos==collaborator

DATASETS=$1
SUFFIX=$2
if [ "$DATASETS" == "tabular" ] || [ "$DATASETS" == "images" ] || [ "$DATASETS" == "images_leave-one-in" ]; then
    module load Julia/1.5.1-linux-x86_64
    module load Python/3.8.2-GCCcore-9.3.0

    julia --threads 48 --project ./generate_stats.jl experiments${SUFFIX}/${DATASETS} evaluation${SUFFIX}/${DATASETS}
    julia --threads 48 --project ./collect_stats.jl  evaluation${SUFFIX}/${DATASETS} evaluation${SUFFIX}/${DATASETS}_eval.bson -f
else
    echo "Unsupported dataset type. Only 'images', 'images_leave-one-in' or 'tabular' are supported."
fi