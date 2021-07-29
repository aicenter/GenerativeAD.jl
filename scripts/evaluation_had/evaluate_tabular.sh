#!/bin/bash
ml Julia
ml Python
julia --project --threads 32 --project ../evaluation/generate_stats.jl experiments/tabular/had_basic evaluation/tabular/had_basic 
julia --project --threads 32 --project ../evaluation/collect_stats.jl evaluation/tabular evaluation/tabular_eval.bson -f
julia --project ./gather_tabular.jl evaluation/tabular_had.bson evaluation/tabular_original.bson evaluation/tabular_eval.bson
julia --project ../evaluation/evaluate_performance.jl evaluation/tabular_eval.bson --output-prefix evaluation/tabular_eval --criterion-metric val_auc --rank-metric tst_auc --backend txt
