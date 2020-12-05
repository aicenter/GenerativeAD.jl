#!/bin/bash
# This runs jacodeco for a certain model over a set of datasets.
# USAGE EXAMPLE
# 	./jacodeco_run_parallel.sh vae tabular datasets_tabular.txt
MODEL=$1 		# which model to run
DATASET_FILE=$2	# file with dataset list
SEED=$3
AC=$4

LOG_DIR="${HOME}/logs/sample_score"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi


while read d; do
	# submit to slurm
    sbatch \
    --output="${LOG_DIR}/${DATATYPE}_${MODEL}_${d}-%A.out" \
    ./sample_score_run.sh $MODEL $d $SEED $AC
done < ${DATASET_FILE}
