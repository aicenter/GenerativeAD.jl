using DrWatson
@quickactivate
using ArgParse
using BSON
using Flux
using GenerativeAD
using StatsBase: fit!, predict, sample

s = ArgParseSettings()
@add_arg_table! s begin
   "max_seed"
		default = 1
		arg_type = Int
		help = "seed"
	"dataset"
		default = "iris"
		arg_type = String
		help = "dataset"
    "contamination"
    	arg_type = Float64
    	help = "contamination rate of training data"
    	default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, contamination = parsed_args

modelname = "sptn"

function sample_params()
	parameter_rng = (
		firstdense = [true, false], 
		batchsize = 2 .^ (5:7), 
		ncomp = 2 .^ (1:4), 
		nlayers = 1:3, 
		activation = [identity], 
		unitary = [:butterfly], 
		sharing = [:dense, :all, :none], 
		init_seed = 1:Int(1e8), 
	)	

	(;zip(keys(parameter_rng), map(x->sample(x, 1)[1], parameter_rng))...)
end


function fit(data, parameters)
	model = GenerativeAD.Models.SPTN(;idim=size(data[1][1],1), parameters...)

	try
		global info, fit_t, _, _, _ = @timed fit!(model, data; max_train_time=82800/max_seed, 
			patience=20, check_interval=10, parameters...)
	catch e
		# return an empty array if fit fails so nothing is computed
		@info "Failed training due to \n$e"
		return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
	end

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		history = info.history,
		npars = info.npars,
		model = info.model
		)

	# now return the different scoring functions
	training_info, [(x -> predict(info.model, x), parameters)]
end

if abspath(PROGRAM_FILE) == @__FILE__
	# set a maximum for parameter sampling retries
	try_counter = 0
	max_tries = 10*max_seed
	cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
	while try_counter < max_tries
	    parameters = sample_params()

	    for seed in 1:max_seed
			savepath = datadir("experiments/tabular$cont_string/$(modelname)/$(dataset)/seed=$(seed)")
			mkpath(savepath)

			# get data
			data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)		
			# edit parameters
			edited_parameters = GenerativeAD.edit_params(data, parameters)
			
			@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
			@info "Train/valdiation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[3][1], 2))"
			@info "Number of features: $(size(data[1][1], 1))"

			# check if a combination of parameters and seed alread exists
			if GenerativeAD.check_params(savepath, edited_parameters)
				# fit
				training_info, results = fit(data, edited_parameters)

				# save the model separately			
				if training_info.model != nothing
					tagsave(joinpath(savepath, savename("model", edited_parameters, "bson", digits=5)), 
						Dict("model"=>training_info.model,
							"fit_t"=>training_info.fit_t,
							"history"=>training_info.history,
							"parameters"=>edited_parameters
							), safe = true)
					training_info = merge(training_info, (model = nothing,))
				end

				# here define what additional info should be saved together with parameters, scores, labels and predict times
				save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, contamination = contamination))

				# now loop over all anomaly score funs
				for result in results
					GenerativeAD.experiment(result..., data, savepath; save_entries...)
				end
				global try_counter = max_tries + 1
			else
				@info "Model already present, trying new hyperparameters..."
				global try_counter += 1
			end
		end
	end
	(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
end
