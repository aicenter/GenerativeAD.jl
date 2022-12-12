#!/bin/bash
for FILE in configs2/*
do
	sbatch ./run_experiment.sh $FILE
done