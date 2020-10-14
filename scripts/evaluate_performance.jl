# run this script as `julia evaluate_performance.jl res.bson` or `julia evaluate_performance.jl dir`
# in the second case, it will recursively search for all compatible files in subdirectories
target = ARGS[1]

using DrWatson
@quickactivate
using EvalMetrics
using FileIO
using BSON
using DataFrames
using ValueHistories
using LinearAlgebra

function compute_stats(f::String)
	data = load(f)
	println(abspath(f))
	#map(p->println("$(p[1]) = $(p[2])"), collect(pairs(data[:parameters])))
	scores_labels = [(data[:val_scores], data[:val_labels]), (data[:tst_scores], data[:tst_labels])]
	setnames = ["validation", "test"]

	results = []
	for (scores, labels) in scores_labels
		scores = vec(scores)
		roc = EvalMetrics.roccurve(labels, scores)
		auc = EvalMetrics.auc_trapezoidal(roc...)
		prc = EvalMetrics.prcurve(labels, scores)
		auprc = EvalMetrics.auc_trapezoidal(prc...)

		t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
		cm5 = ConfusionMatrix(labels, scores, t5)
		tpr5 = EvalMetrics.true_positive_rate(cm5)
		f5 = EvalMetrics.f1_score(cm5)

		push!(results, [auc, auprc, tpr5, f5])
	end

	DataFrame(measure = ["AUC", "AUPRC", "TPR@5", "F1@5"], validation = results[1], test = results[2])
end

function query_stats(target::String)
	if isfile(target)
		try
			println(compute_stats(target))
		catch e
			@info "$target not compatible"
		end
	else
		query_stats.(joinpath.(target, filter(x->!(occursin("model_", string(x))), readdir(target))))
	end
end

query_stats(target)
