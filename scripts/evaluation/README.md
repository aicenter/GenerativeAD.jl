# Evaluation scripts
Due to the large number of files that need to be processed the main evaluation is split into three steps
- generation of files with statistics (parallel) - `generate_stats.jl`
- collection of previously generated files into one cached dataframe (parallel) - `collect_stats.jl`
- printing of summary tables - `evaluate_performance.jl`

The first two steps can be run as a batch job by running `sbatch run_eval.sh` script. By default this won't rewrite any precomputed files with the exception of the cached DataFrame in the second step.

```
    julia --threads 16 --project ./generate_stats.jl experiments/images evaluation/images 
    julia --threads 16 --project ./collect_stats.jl  evaluation/images evaluation/images_eval.bson -f
```

The third step is intended to be run in more interactive manner as it allows multiple summary options.
- loading `evaluation/images_eval.bson` cache, using `val_auc` for sorting models and `tst_auc` for final ranking, storing the rank table as `html` page in data prefixed folder given by `--output-prefix`
```
    julia --project ./evaluate_performance.jl \
                        evaluation/images_eval.bson \
                        --output-prefix evaluation/images_eval \
                        --criterion-metric val_auc \
                        --rank-metric tst_auc \
                        --backend html 
```

- loading `evaluation/images_eval.bson` cache, using `val_pat_10` for sorting models and `tst_pat_10` for final ranking, printing the result to stdout, additionally `--best-params` will store metadata for best models of each type into separate CSVs
```
    julia --project ./evaluate_performance.jl \
                        evaluation/images_eval.bson \
                        --criterion-metric val_pat_10 \
                        --rank-metric tst_pat_10 \
                        --backend txt \
                        --verbose \
                        --best-params
```

- loading `evaluation/images_eval.bson` cache, using `val_pat_x` with increasing `x` for sorting models and `tst_pat_10` for final ranking, printing the ranking given percentage of labeled samples in the validation to txt file in data prefixed folder given by `--output-prefix`
```
    julia --project ./evaluate_performance.jl \
                        evaluation/images_eval.bson \
                        --output-prefix evaluation/images_eval \
                        --rank-metric tst_pat_10 \
                        --proportional
```

Some combination of parameters don't make sense, such as running with `--best-params` while also using `--proportional`. Furthermore the tex(latex) output does not escape underscores and therefore cannot be parsed sometimes.