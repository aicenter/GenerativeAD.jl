#!/bin/bash
for FILE in configs4/*
do
	sbatch ./run_experiment.sh $FILE
done