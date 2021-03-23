struct AE
    encoder
    decoder
end

Flux.@functor AE

function (ae::AE)(x)
    z = ae.encoder(x)
    x̂ = ae.decoder(z)
    return x̂
end

# ae_loss already in svdd

function conv_ae_constructor(
	;idim=(2,2,1), 
	zdim::Int=1, 
	activation="relu", 
	kernelsizes=(1,1), 
	channels=(1,1), 
	scalings=(1,1),
	init_seed=nothing, 
	batchnorm=false, 
	kwargs...
)
	# if seed is given, set it
	(init_seed !== nothing) ? Random.seed!(init_seed) : nothing
	# encoder 
	encoder = Chain(
		conv_encoder(idim, zdim, kernelsizes, channels, scalings; activation=activation, batchnorm=batchnorm)...
	)
	# decoder 
	decoder = Chain(
		conv_decoder(idim, zdim, reverse(kernelsizes), reverse(channels), reverse(scalings),
			activation=activation, vec_output=false, vec_output_dim=nothing, batchnorm=batchnorm)...,
			x->sigmoid.(x) # our dataset images are scaled from 0-1
	)
	# reset seed
	(init_seed !== nothing) ? Random.seed!() : nothing

	model = AE(encoder, decoder)
end


function ae_constructor(
	;idim::Int=1, 
	zdim::Int=1, 
	hdim::Int=64, 
	activation = "relu", 
	nlayers::Int=3, 
	init_seed=nothing, 
	kwargs...
	)
	(init_seed !== nothing) ? Random.seed!(init_seed) : nothing

	encoder = build_mlp(idim, hdim, zdim, nlayers, activation=activation, lastlayer="linear")

	decoder = build_mlp(zdim, hdim, idim, nlayers, activation=activation, lastlayer="linear")

	(init_seed !== nothing) ? Random.seed!() : nothing

	return AE(encoder, decoder)
end

function anomaly_score(ae::AE, real_input; dims=(1,2,3), to_testmode::Bool=true)
    (to_testmode == true) ? Flux.testmode!(ae) : nothing
    score = vec(Flux.sum((ae(real_input) .- real_input).^2, dims=dims))
    (to_testmode == true) ? Flux.testmode!(ae, false) : nothing
	return scores
end

function anomaly_score_gpu(ae::AE, real_input; batch_size=64, dims=(1,2,3), to_testmode::Bool=true)
    real_input = Flux.Data.DataLoader(real_input, batchsize=batch_size)
	(to_testmode == true) ? Flux.testmode!(ae) : nothing
	ae = ae |> gpu
	scores = Array{Float32}([])
	for x in real_input
		x = x |> gpu
		score = vec(Flux.sum((ae(x) .- x).^2, dims=dims))
		scores = cat(scores, score |> cpu, dims=1)
	end
	(to_testmode == true) ? Flux.testmode!(ae, false) : nothing
	return scores
end


function StatsBase.fit!(ae::AE, data, params)
	# prepare batches & loaders
	train_loader, val_loader = prepare_dataloaders(data, params)
	history = MVHistory()
	# prepare for early stopping
	best_val_loss = Inf
	
	patience = params.patience
	val_batches = length(val_loader)

    if ndims(first(train_loader)) == 2
		dims = 1
	elseif ndims(first(train_loader)) == 4
		dims = (1,2,3)
	else 
		error("unknown data type (no 2D or 4D tensor)")
	end

	#model to gpu
	ae = ae |> gpu
	# ADAMW(η = 0.001, β = (0.9, 0.999), decay = 0) = Optimiser(ADAM(η, β), WeightDecay(decay))
	opt_ae = haskey(params, :decay) ? ADAMW(params.lr, (0.9, 0.999), params.decay) : ADAM(params.lr)

	best_ae = deepcopy(ae)

	# train "AE"
	progress = Progress(params.iters)
    ps_ae = Flux.params(AE)

	for (iter, X) in enumerate(train_loader)
		X = getobs(X)|>gpu
		loss_ae, back = Flux.pullback(ps_ae) do
			ae_loss(ae(X), X, dims=dims)
		end
		grad_ae = back(1f0)
		Flux.Optimise.update!(opt_ae, ps_ae, grad_ae)


        push!(history, :loss_ae, iter, loss_ae)

		next!(progress; showvalues=[
			(:iters, "$(iter)/$(params.iters)"),
			(:ae_loss, loss_ae)
			])

		if mod(iter, params.check_every) == 0
			total_val_loss = 0
			Flux.testmode!(ae)
			for X_val in val_loader
				X_val = X_val |> gpu
				total_val_loss += ae_loss(ae(X), X)
			end
			Flux.testmode!(ae, false)
			push!(history, :validation_ae_loss, iter, total_val_loss/val_batches)
			if total_val_loss < best_val_loss
				best_val_loss = total_val_loss
				patience = params.patience
				best_ae = deepcopy(ae)
			else
				patience -= 1
				if patience == 0
					@info "Stopped training of Autoencoder after $(iter) iterations"
                    global iters = iter - params.check_every*params.patience
					break
				end
			end
		end  
        if iter == params.iters
			global iters = params.iters
		end      
	end
	return history, best_ae, sum(map(p->length(p), Flux.params(ae))), iters
end
