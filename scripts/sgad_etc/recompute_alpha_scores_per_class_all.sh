#!/bin/bash
METHOD=$1
BASE_BETA=$2
FORCE=$3

for ANOMALY_CLASS in {1..10}
do
	./recompute_alpha_scores_per_class_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in knn ${ANOMALY_CLASS} ${METHOD} ${BASE_BETA} $FORCE
	./recompute_alpha_scores_per_class_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal ${ANOMALY_CLASS} ${METHOD} ${BASE_BETA} $FORCE
	./recompute_alpha_scores_per_class_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in kld ${ANOMALY_CLASS} ${METHOD} ${BASE_BETA} $FORCE
done