using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using GenerativeAD.Models
using BSON
using StatsBase: fit!, predict, sample

using Flux
using MLDataPattern

s = ArgParseSettings()
@add_arg_table! s begin
	"max_seed"
		default = 1
		arg_type = Int
		help = "max_seed"
	"category"
		default = "wood"
		arg_type = String
		help = "category"
	"contamination"
		arg_type = Float64
		help = "contamination rate of training data"
		default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack category, max_seed, contamination = parsed_args

modelname ="DeepSVDD"

function sample_params()
	# first sample the number of layers
	nlayers = rand(2:4)
	kernelsizes = reverse((3,5,7,9)[1:nlayers])
	channels = reverse((16,32,64,128)[1:nlayers])
	scalings = reverse((1,2,2,2)[1:nlayers])
	
	par_vec = (
		2 .^(3:8), 
		10f0 .^(-4:-3), 
		10f0 .^(-4:-3),
		[true, false], 
		2 .^ (3:5), 
		["relu", "swish", "tanh"], 
		["soft-boundary", "one-class"],
		[0.01f0, 0.1f0, 0.5f0, 0.99f0], # paper 0.1
		[1e-6], # from paper
		1:Int(1e8)
	)
	argnames = (
		:zdim, 
		:lr_ae, 
		:lr_svdd, 
		:batchnorm,
		:batch_size, 
		:activation, 
		:objective,
		:nu,
		:decay,
		:init_seed
	)
	parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
	return merge(parameters, (nlayers=nlayers, kernelsizes=kernelsizes,
		channels=channels, scalings=scalings))
end


function fit(data, parameters)
	# define models (Generator, Discriminator)
	all_parameters = merge(
		parameters, 
		(
			idim=size(data[1][1])[1:3], 
			iters=5000, 
			check_every=30, 
			ae_check_every = 200,
			patience=10, 
			ae_iters=10000
		)
	)
	svdd = GenerativeAD.Models.conv_svdd_constructor(;all_parameters...) |> gpu

	# define optimiser
	try
		global info, fit_t, _, _, _ = @timed fit!(svdd, data, all_parameters)
	catch e
		println("Error caught => $(e).")
		return (fit_t = NaN, model = nothing, history = nothing, n_parameters = NaN), []
	end

	training_info = (
		fit_t = fit_t,
		model = info[2]|>cpu,
		history_svdd = info[1][1], # losses through time
		history_ae = info[1][2],
		npars = info[3], # number of parameters
		iters = info[4] # optim iterations of model
		)

	return training_info, [(x -> GenerativeAD.Models.anomaly_score_gpu(svdd, x), parameters)]
end

#_________________________________________________________________________________________________

if abspath(PROGRAM_FILE) == @__FILE__
	try_counter = 0
	max_tries = 10*max_seed
	cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
	while try_counter < max_tries
		parameters = sample_params()

		for seed in 1:max_seed
			savepath = datadir("experiments/images_mvtec$cont_string/$(modelname)/$(category)/ac=1/seed=$(seed)")
			mkpath(savepath)

			# get data
			data = GenerativeAD.load_data("MVTec-AD", seed=seed, category=category, 
					contamination=contamination, img_size=128)

			@info "Trying to fit $modelname on $category with parameters $(parameters)..."
			@info "Train/validation/test splits: $(size(data[1][1], 4)) | $(size(data[2][1], 4)) | $(size(data[3][1], 4))"
			@info "Number of features: $(size(data[1][1])[1:3])"

			# check if a combination of parameters and seed alread exists
			if GenerativeAD.check_params(savepath, parameters)
				# fit
				training_info, results = fit(data, parameters)

				# save the model separately
				if training_info.model !== nothing
					tagsave(joinpath(savepath, savename("model", parameters, "bson", digits=5)), 
						Dict("model"=>training_info.model,
							 "fit_t"=>training_info.fit_t
						), 
						safe = true)
					training_info = merge(training_info, (model = nothing,))
				end

				# here define what additional info should be saved together with parameters, scores, labels and predict times
				save_entries = merge(training_info, (modelname = modelname, seed = seed,
					category = category,
					contamination=contamination))

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