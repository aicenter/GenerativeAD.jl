#!/bin/bash
# This runs jacodeco for a certain model over a set of datasets.
# USAGE EXAMPLE
# 	./jacodeco_run_parallel.sh vae tabular datasets_tabular.txt
MODEL=$1 		# which model to run
DATATYPE=$2			# tabular/image
DATASET_FILE=$3	# file with dataset list

LOG_DIR="${HOME}/logs/jacodeco"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read d; do
	# submit to slurm
    sbatch \
    --output="${LOG_DIR}/${DATATYPE}_${MODEL}_${d}-%A.out" \
    ./jacodeco_run.sh $MODEL $DATATYPE $d
done < ${DATASET_FILE}
