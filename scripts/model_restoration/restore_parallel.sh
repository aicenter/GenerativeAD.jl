#!/bin/bash
# This regenerates the modelfiles for model and specific data type
# Run from this folder only.
MODEL=$1 	# which model to run
DATA=$2		# tabular/images
DATASET_FILE=$3	# file with dataset list
FORCE=$4        # force overwrite

LOG_DIR="${HOME}/logs/${MODEL}_rest"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read d; do
	# submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}-%A.out" \
	./restore_modelfiles.sh /home/skvarvit/generativead/GenerativeAD.jl/data/experiments/${DATA}/${MODEL}/${d} $FORCE
done < ${DATASET_FILE}
