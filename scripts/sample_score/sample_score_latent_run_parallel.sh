#!/bin/bash
# This runs jacodeco for a certain model over a set of datasets.
# USAGE EXAMPLE
# 	./jacodeco_run_parallel.sh vae tabular datasets_tabular.txt
MODEL=$1 		# which model to run
DATATYPE=$2			# tabular/image
DATASET_FILE=$3	# file with dataset list
SEED=$4
AC=$5

LOG_DIR="${HOME}/logs/sample_score"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi


while read d; do
	if [ $d = "MNIST" ] || [ $d = "FashionMNIST" ] || [ $d = "CIFAR10" ] || [ $d = "SVHN2" ]
	then
		RUNSCRIPT="./sample_score_latent_gpu_run.sh"
	else
		RUNSCRIPT="./sample_score_latent_run.sh"
	fi

	# submit to slurm
    sbatch \
    --output="${LOG_DIR}/${DATATYPE}_${MODEL}_${d}-%A.out" \
    $RUNSCRIPT $MODEL $DATATYPE $d $SEED $AC
done < ${DATASET_FILE}
