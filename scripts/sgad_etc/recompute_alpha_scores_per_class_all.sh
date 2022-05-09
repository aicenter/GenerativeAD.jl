#!/bin/bash
METHOD=$1
FORCE=$2

for ANOMALY_CLASS in {1..10}
do
	./recompute_alpha_scores_per_class_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in knn ${ANOMALY_CLASS} ${METHOD} $FORCE
done