struct adVAE
	encoder
	generator
	transformer
end

function adVAE(;idim::Int=1, zdim::Int=1, hdim::Int=64, activation = "relu", nlayers::Int=3, kwargs...)
	encoder = Flux.Chain(
		build_mlp(idim, hdim, hdim, nlayers-1, activation=activation),
		ConditionalDists.SplitLayer(hdim, [zdim,zdim], [identity,softplus])
	)
	transformer = Flux.Chain(
		build_mlp(zdim*2, hdim, hdim, 2, activation=activation),
		ConditionalDists.SplitLayer(hdim, [zdim,zdim], [identity,softplus])
	)
	generator = build_mlp(zdim, hdim, idim, nlayers, activation=activation, lastlayer="linear")
	
	return adVAE(encoder, generator, transformer)
end

function Conv_adVAE(
		;isize::Int=1, 
		in_ch::Int=1, 
		zdim::Int=64, 
		nf::Int=16, 
		extra_layers::Int=0, 
		hdim::Int=64,
		depth::Int=3, 
		activation="relu", 
		kwargs...
		)

	encoder = Flux.Chain(
		ConvEncoder(isize, in_ch, zdim, nf, extra_layers),
		x->reshape(x, (zdim,size(x)[end])), # equivalent to flatten
		ConditionalDists.SplitLayer(zdim, [zdim, zdim], [identity, softplus])
	)
	transformer = Flux.Chain(
		build_mlp(zdim*2, hdim, hdim, depth, activation=activation),
		ConditionalDists.SplitLayer(hdim, [zdim,zdim], [identity,softplus])
	)
	generator = Flux.Chain(
		Flux.Dense(zdim, zdim, eval(:($(Symbol(activation)))) ),
		x->reshape(x, (1,1,zdim,size(x)[end])), # inverse to flatten
		ConvDecoder(isize, zdim, in_ch, nf, extra_layers) 
	)

	return adVAE(encoder, generator, transformer)
end

Flux.@functor adVAE

function (advae::adVAE)(x)
	μ, Σ = advae.encoder(x)
	z = μ + Σ * randn(Float32)
	xᵣ = advae.generator(z)
end


kl_divergence(μ, Σ) = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σ.^2) - μ.^2  - Σ.^2, dims=1)) 
kl_divergence(μ₁, Σ₁, μ₂, Σ₂) = Flux.mean(sum(log.(Σ₂) - log.(Σ₁) + (Σ₁.^2 + (μ₁ - μ₂).^2) ./ (2*Σ₂.^2) .- 0.5f0, dims=1))

function loss(advae::adVAE, x; γ=1e-3, λ=1e-2, mx=1, mz=1)
	μ, Σ = advae.encoder(x)
	μₜ, Σₜ = advae.transformer(cat(μ, Σ, dims=1))

	z = μ + Σ * randn(Float32)
	zₜ = μₜ + Σₜ * randn(Float32)

	xᵣ = advae.generator(z)
	xₜᵣ = advae.generator(zₜ)
	
	μᵣ, Σᵣ = advae.encoder(xᵣ)
	μₜᵣ, Σₜᵣ = advae.encoder(xₜᵣ)

	# 𝓛 for generator => 𝓛 = 𝓛_z + 𝓛_zₜ 
	𝓛_z = Flux.Losses.mse(x, xᵣ) .+ γ * kl_divergence(μᵣ, Σᵣ)
	𝓛_zₜ = max(0f0, (mx - Flux.Losses.mse(xᵣ, xₜᵣ))) + γ * max(0f0, (mz - kl_divergence(μₜᵣ, Σₜᵣ)))
	𝓛 = 𝓛_z + 𝓛_zₜ
	# 𝓛ₜ for transformer 
	𝓛ₜ = kl_divergence(μ, Σ, μₜ, Σₜ)
	# 𝓛ₑ for encoder
	𝓛ₑ = Flux.Losses.mse(x, xᵣ) .+ λ * kl_divergence(μ, Σ) 
		+ γ * max(0f0, (mz - kl_divergence(μᵣ, Σᵣ)))
		+ γ * max(0f0, (mz - kl_divergence(μₜᵣ, Σₜᵣ)))

	return 𝓛 + λ*𝓛ₜ, 𝓛ₑ, 𝓛, 𝓛ₜ
end


function anomaly_score(advae::adVAE, real_input; L=100, dims=3, batch_size=64, to_testmode::Bool=true)
	real_input = Flux.Data.DataLoader(real_input, batchsize=batch_size)
	(to_testmode == true) ? Flux.testmode!(advae) : nothing
	advae = advae |> gpu
	output = Array{Float32}([])
	for x in real_input
        x = x |> gpu
		X = Array{Float32}(undef, size(x)[end], 1)
		μ, Σ = advae.encoder(x)
		for l=1:L
			z = μ + Σ * randn(Float32)
			xᵣ = advae.generator(z)
			rec_loss = vec(Flux.mse(x, xᵣ, agg=x->Flux.mean(x, dims=dims))) |> cpu # loss per batch
			X = cat(X, rec_loss, dims=2)
		end
		output = cat(output, Flux.mean(X, dims=2), dims=1)
	end
	(to_testmode == true) ? Flux.testmode!(advae, false) : nothing
	return output
end


function StatsBase.fit!(advae::adVAE, data, params)
	# prepare batches & loaders
	train_loader, val_loader = prepare_dataloaders(data, params)
	# training info logger
	history = MVHistory()
	# prepare for early stopping
	best_adVAE = deepcopy(advae)
	patience = params.patience
	best_val_loss = 1e10
	val_batches = length(val_loader)

	# ADAMW(η = 0.001, β = (0.9, 0.999), decay = 0) = Optimiser(ADAM(η, β), WeightDecay(decay))
	opt_step1 = haskey(params, :decay) ? ADAMW(params.lr, (0.9, 0.999), params.decay) : ADAM(params.lr)
	opt_step2 = haskey(params, :decay) ? ADAMW(params.lr, (0.9, 0.999), params.decay) : ADAM(params.lr)

	ps_step1 = Flux.params(advae.generator, advae.transformer)
	ps_step2 = Flux.params(advae.encoder)

	progress = Progress(length(train_loader))
	for (iter, X) in enumerate(train_loader)
		# training step 1
		loss_1, back = Flux.pullback(ps_step1) do
			loss(advae|>gpu , getobs(X)|>gpu, γ=params.gamma, λ=params.lambda, mx=params.mx, mz=params.mz)
		end
		grad_step1 = back((1f0, 0f0, 0f0, 0f0))
		Flux.Optimise.update!(opt_step1, ps_step1, grad_step1)
		# training step 2
		loss_2, back = Flux.pullback(ps_step2) do
			loss(advae|>gpu , getobs(X)|>gpu, γ=params.gamma, λ=params.lambda, mx=params.mx, mz=params.mz)
		end
		grad_step2 = back((0f0, 1f0, 0f0, 0f0))
		Flux.Optimise.update!(opt_step1, ps_step2, grad_step2)


		push!(history, :loss_step_1, iter, loss_1[1])
		push!(history, :encoder_loss, iter, loss_2[2])
		push!(history, :generator_loss, iter, loss_1[3])
		push!(history, :transformer_loss, iter, loss_1[4])

		next!(progress; showvalues=[
			(:iters, "$(iter)/$(params.iters)"),
			(:loss_step_1, loss_1[1]),
			(:loss_step_2, loss_2[2])
			])
		# TODO: check and prepare early stopping 
		if mod(iter, params.check_every) == 0
			total_val_loss = 0
			Flux.testmode!(advae)
			for X_val in val_loader
				X_val = X_val |> gpu
				μ, Σ = advae.encoder(X_val)
				z = μ + Σ * randn(Float32)
				xᵣ = advae.generator(z)
				total_val_loss += Flux.mse(X_val, xᵣ)
			end
			Flux.testmode!(advae, false)
			push!(history, :validation_loss, iter, total_val_loss/val_batches)

			if total_val_loss < best_val_loss
				best_val_loss = total_val_loss
				patience = params.patience
				best_adVAE = deepcopy(advae)
			else
				patience -= 1
				if patience == 0
					@info "Stopped training after $(iter) iterations"
					global iters = iter - params.check_every*params.patience
					break
				end
			end
		end
		if iter == params.iters
			global iters = params.iters
		end
	end
	return history, best_adVAE, sum(map(p->length(p), Flux.params(advae))), iters
end
