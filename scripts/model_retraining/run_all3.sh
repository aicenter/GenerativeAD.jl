#!/bin/bash
for FILE in configs3/*
do
	sbatch ./run_experiment.sh $FILE
done