using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux
using MLDataPattern
using IPMeasures
using GenerativeModels
using Distributions
using LinearAlgebra
using ValueHistories
using FileIO
using DistributionsAD

"""
	save_results(parameters, training_info, results, savepath, data, ac)

this computes and saves score and model files
"""
function save_results(parameters, training_info, results, savepath, data, ac)
	# save the model separately			
	tagsave(joinpath(savepath, savename("model", parameters, "bson", digits=5)), 
		Dict("model"=>training_info.model,
			 "tr_encodings"=>training_info.tr_encodings,
			 "val_encodings"=>training_info.val_encodings,
			 "tst_encodings"=>training_info.tst_encodings,
			 "fit_t"=>training_info.fit_t,
			 "history"=>training_info.history,
			 "parameters"=>parameters
			 ), 
		safe = true)
	training_info = merge(training_info, 
		(model=nothing,tr_encodings=nothing,val_encodings=nothing,tst_encodings=nothing))

	# here define what additional info should be saved together with parameters, scores, labels and predict times
	save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, 
		anomaly_class = ac,
		contamination=contamination))

	# now loop over all anomaly score funs
	for result in results
		GenerativeAD.experiment(result..., data, savepath; save_entries...)
	end
end
