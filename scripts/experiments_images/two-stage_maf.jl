using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
# because of vae and 2stage
using DataFrames
using CSV
using ValueHistories
using Flux
using ConditionalDists
using GenerativeModels
import GenerativeModels: VAE
using Distributions
using DistributionsAD

s = ArgParseSettings()
@add_arg_table! s begin
   "max_seed"
		required = true
		arg_type = Int
		help = "seed"
	"dataset"
		required = true
		arg_type = String
		help = "dataset"
	"tab_name"
		required = true
		arg_type = String
		help = "name of tab -> example: vae_LOSS_images, wae-vamp_AUC_images"
	"anomaly_classes"
		arg_type = Int
		default = 10
		help = "number of anomaly classes"
	"mi_only"
		arg_type = Int
		default = -1
		help = "index of model in range 1 to 10 or -1 for all models"
	"method"
		arg_type = String
		default = "leave-one-out"
		help = "method for data creation -> \"leave-one-out\" or \"leave-one-in\" "
    "contamination"
    	arg_type = Float64
    	help = "contamination rate of training data"
    	default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, tab_name, anomaly_classes, mi_only, method, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################

sp = split(tab_name, "_")
enc = sp[1]
criterion = lowercase(sp[2])

modelname = "$(enc)_maf"

function sample_params()
	parameters_rng = (
		nflows 		= 2 .^ (1:3),
		hdim 		= 2 .^(4:10),
		nlayers 	= 2:3,
		ordering 	= ["natural", "random"],
		lr 			= [1f-4],
		batchsize 	= 2 .^ (5:7),
		act_loc		= ["relu", "tanh"],
		act_scl		= ["relu", "tanh"],
		bn 			= [true, false],
		wreg 		= [0.0f0, 1f-5, 1f-6],
		init_I 		= [true, false],
		init_seed 	= 1:Int(1e8)
	)
	
	return (;zip(keys(parameters_rng), map(x->sample(x, 1)[1], parameters_rng))...)
end

function fit(data, parameters, aux_info)
	model = GenerativeAD.Models.MAF(;idim=size(data[1][1], 1), parameters...)

	try
		global info, fit_t, _, _, _ = @timed fit!(model, data; 
			max_train_time=82800/max_seed/anomaly_classes/2/length(mi_only), 
					patience=10, check_interval=50, parameters...)
	catch e
		@info "Failed training due to \n$e"
		return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), []
	end

	training_info = (
		fit_t = fit_t,
		history = info.history,
		niter = info.niter,
		npars = info.npars,
		model = info.model
		)

	training_info, [(x -> predict(info.model, x), parameters)]
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# set a maximum for parameter sampling retries
try_counter = 0
max_tries = 10*max_seed
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
while try_counter < max_tries
	parameters = sample_params()

	for seed in 1:max_seed
		for i in 1:anomaly_classes
			savepath = datadir("experiments/images_$(method)$cont_string/$(modelname)/$(dataset)/ac=$(i)/seed=$(seed)")
			mi_indexes = (mi_only == -1) ? [1:10...] : [mi_only] 
			for mi = mi_indexes
				aux_info = (model_index=mi, criterion=criterion)

				data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=i, method=method, contamination=contamination)
				data, encoding_name, encoder_params = GenerativeAD.Models.load_encoding(tab_name, data, i, dataset=dataset, seed=seed, model_index=mi)
				edited_parameters = GenerativeAD.edit_params(data, parameters)

				@info "Started training $(modelname)$(edited_parameters) on $(dataset):$(seed)"
				@info "Train/valdiation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[2][1], 2))"
				@info "Number of features: $(size(data[1][1], 1))"
			
				# here, check if a model with the same parameters was already tested
				@info "Trying to fit $modelname on $dataset with parameters $(parameters)..."
				if GenerativeAD.check_params(savepath, merge(edited_parameters, aux_info))
					training_info, results = fit(data, edited_parameters, aux_info);
					# here define what additional info should be saved together with parameters, scores, labels and predict times
					if training_info.model !== nothing
						tagsave(joinpath(savepath, savename("model", edited_parameters, "bson", digits=5)), 
								Dict("model"=>training_info.model,
									"fit_t"=>training_info.fit_t,
									"history"=>training_info.history,
									"parameters"=>edited_parameters
									), safe = true)
						training_info = merge(training_info, (model = nothing,))
					end
					save_entries = merge(training_info, (
						modelname = modelname, 
						seed = seed, 
						dataset = dataset, 
						anomaly_class = i, 
						encoder=encoding_name,
						encoder_params=encoder_params,
						model_index=mi,
						criterion=criterion,
						contamination=contamination))
					# now loop over all anomaly score funs
					for result in results
						GenerativeAD.experiment(result..., data, savepath; save_entries...)
					end
					global try_counter = max_tries + 1
				else
					@info "Model already present, sampling new hyperparameters..."
					global try_counter += 1
				end
			end
		end
	end
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
