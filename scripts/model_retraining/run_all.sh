#!/bin/bash
for FILE in sgvaegan100_configs/*
do
	sbatch ./run_experiment.sh $FILE
done