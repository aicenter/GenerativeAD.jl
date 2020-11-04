using DrWatson
using DataFrames
using CSV
using BSON
using ValueHistories
# because of vae
using GenerativeAD
using GenerativeAD.Models
using ConditionalDists
using GenerativeModels
import GenerativeModels: VAE
using Distributions
using DistributionsAD


"""
	helper functions for two stage models
"""
function string2dict(params)
	function get_values(value)
		try 
			return parse(Int64, value)
		catch e
			if occursin("f0", value)
				return parse(Float32, value[1:end-2])
			else
				return value[2:end-1]
			end
		end
	end

	params = map(x->split(x, " = "), split(params, ","))
	key = map(x->x[1][2:end], params)
	value = map(x->get_values(x[2]), params)
	return Dict(map(x->x[1]=>x[2], zip(key, value)))
end


function return_best_n(df, rev=false, n=10, dataset="MNIST")
	df = df[df.dataset .== dataset, :]
	top = first(sort(by(df, :params, :criterion => mean), :criterion_mean, rev=rev), n)
	df_top = []
	for i=1:n
		tmp = df[df.params .== top.params[i],:]
		ind = i.*ones(size(tmp,1))
		push!(df_top, hcat(tmp, DataFrame(ind=ind)))
	end
	return vcat(df_top...)
end


function load_encoding(model, data; dataset::String="iris", seed::Int=1, model_index::Int=1)
	# model = "vae_AUC_tabular"
	# load csv
	df = CSV.read(datadir("$(model)_best_tab.csv")) 
	df = df[df.dataset .== dataset, :]
	#df = return_best_n(df, 10, dataset)
	# checking if model configuration was trained on all classes and all seeds
	n_comb = 50
	check_comb = sum([sum(df.ind .== i) for i=1:10])
	if check_comb < n_comb
		@info "One of chosen models does not include all seeds!! $(check_comb) out of $(n_comb)"
	end
	# get correct model path
	encoding_path = df[(df.seed .==seed) .& (df.ind .==model_index),:][:path][1]
	encoder_params = string2dict(df[(df.seed .==seed) .& (df.ind .==model_index),:][:params][1])
	# load model and encode data
	model = BSON.load(encoding_path)
	encodings = map(x->cpu(GenerativeAD.Models.encode_mean(model["model"], x)), (data[1][1], data[2][1], data[3][1]))
	data = (encodings[1], data[1][2]), (encodings[2], data[2][2]), (encodings[3], data[3][2])

	return data, split(encoding_path, "/")[end], encoder_params, model["fit_t"]
end


function load_encoding(model, data, anomaly_class; dataset::String="MNIST", seed::Int=1, model_index::Int=1)
	# model ="vae_AUC_images"
	# load csv
	df = CSV.read(datadir("$(model)_best_tab.csv")) 
	#df = return_best_n(df, 10, dataset)
	df = df[df.dataset .== dataset, :]
	# checking if model configuration was trained on all classes and all seeds
	n_comb = 100
	check_comb = sum([sum(df.ind .== i) for i=1:10])
	if check_comb < n_comb
		@info "One of chosen models does not include all anomaly classes and all seeds!! $(check_comb) out of $(n_comb)"
	end
	# get correct model path
	encoding_path = df[(df.ac .== anomaly_class) .& (df.seed .==seed) .& (df.ind .==model_index),:][:path][1]
	encoder_params = string2dict(df[(df.ac .== anomaly_class) .& (df.seed .==seed) .& (df.ind .==model_index),:][:params][1])
	# load model and encodings
	model = BSON.load(encoding_path)
	data = (model["tr_encodings"], data[1][2]), (model["val_encodings"], data[2][2]), ( model["tst_encodings"], data[3][2])

	return data, split(encoding_path, "/")[end], encoder_params, model["fit_t"]
end
