#!/bin/bash
FORCE=$1

./regenerate_stats_parallel_original.sh $FORCE
./regenerate_stats_parallel_new.sh $FORCE