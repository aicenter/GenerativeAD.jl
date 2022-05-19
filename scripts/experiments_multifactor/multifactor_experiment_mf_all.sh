#!/bin/bash

DATASET=$1

# one factor
./multifactor_experiment_mf_parallel.sh models.txt $DATASET 1
./multifactor_experiment_mf_parallel.sh models.txt $DATASET 2
./multifactor_experiment_mf_parallel.sh models.txt $DATASET 3

# two factors
./multifactor_experiment_mf_parallel.sh models.txt $DATASET 1 2
./multifactor_experiment_mf_parallel.sh models.txt $DATASET 1 3
./multifactor_experiment_mf_parallel.sh models.txt $DATASET 2 3

# three factors
./multifactor_experiment_mf_parallel.sh models.txt $DATASET 1 2 3