using ArgParse
using DrWatson
@quickactivate
using GenerativeAD

s = ArgParseSettings()
@add_arg_table! s begin
    "exp_prefix"
		arg_type = String
		default = "experiments/tabular"
		help = "Data prefix of experiment files."
	"eval_prefix"
		arg_type = String
		default = "evaluation/tabular"
		help = "Data prefix of evaluation files."
	"-d", "--dry_run"
    	action = :store_true
		help = "Compute diff but do not delete."
end

"""
	synchronize_stats(exp_prefix::String, eval_prefix::String; dry_run=false)

Collects filesnames from evaluation datadir in `eval_prefix` and compares it to
contents of experiments in `exp_prefix` datadir. If `dry_run` is true the function does
not delete the evaluation files for which there are no longer experiment files.
"""
function synchronize_stats(exp_prefix::String, eval_prefix::String; dry_run=false)
    exp_dir = datadir(exp_prefix)
    eval_dir = datadir(eval_prefix)
    
    @info "Collecting files from $exp_dir and $eval_dir folders."
    exp_files = GenerativeAD.Evaluation.collect_files(exp_dir)
    filter!(x -> !startswith(basename(x), "model"), exp_files)
    eval_files = GenerativeAD.Evaluation.collect_files(eval_dir)
    
    @info "Collected $(length(exp_files)) files from experiment folder."
    @info "Collected $(length(eval_files)) files from evaluation folder."
    
    # reverse naming of exp files to evaluation
    exp_eval_files = joinpath.(dirname.(exp_files), "eval_" .* basename.(exp_files))
    exp_eval_files = replace.(exp_eval_files, exp_prefix => eval_prefix)

    dif = setdiff(eval_files, exp_eval_files)
    @info "$(length(dif)) evaluation files lack underlying experiment files."
    
    if !dry_run
        @info "Removing files."
        rm.(dif)
    end
end


function main(args)
    @unpack exp_prefix, eval_prefix, dry_run = args
	synchronize_stats(exp_prefix, eval_prefix; dry_run=dry_run)
	@info "---------------- DONE -----------------"
end

main(parse_args(ARGS, s))
