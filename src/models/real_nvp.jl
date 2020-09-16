using Flux
using StatsBase
using Distributions
using DistributionsAD
using ContinuousFlows: RealNVP
using ValueHistories
using MLDataPattern: RandomBatches

struct RealNVPFlow
	flows
	base
end

function RealNVPFlow(nflows::Int, isize::Int, hsize::Int, nlayers::Int)
	RealNVPFlow(Chain([
		RealNVP(
			isize, 
			(d, o, ftype, postprocess) -> build_mlp(d, hsize, o, nlayers, ftype=ftype, lastlayer=postprocess),
			mod(i,2) == 0)
		for i in 1:nflows]...), MvNormal(isize, 1.0f0))
end

(nvpf::RealNVPFlow)(X) = nvpf.flows(X)

Flux.trainable(nvpf::RealNVPFlow) = (nvpf.flows, )

function loss(model::RealNVPFlow, X)
    Z, logJ = model((X, _init_logJ(X)))
    -sum(logpdf(model.base, Z)' .+ logJ)/size(X, 2)
end

function StatsBase.fit!(model::RealNVPFlow, data::Tuple, p)
	opt = Flux.ADAM(p.lr)
	
	trn_model = deepcopy(model)
	ps = Flux.params(trn_model);

	X = data[1][1]
	X_val = data[2][1]

	train_step = 0
	history = MVHistory()
	patience = p.patience
	reg = (p.wreg > 0) ? l2_reg : _ -> 0.0
	
	best_val_loss = loss(trn_model, X_val)
	i = 1
	for batch in RandomBatches(X, p.batchsize)
		l = 0.0f0
		gs = gradient(() -> begin l = loss(trn_model, batch) + p.wreg*reg(ps) end, ps)
		Flux.update!(opt, ps, gs)

		train_step += 1
		
		# validation/early stopping
		val_loss = loss(trn_model, X_val)
		@info "$i - loss: $l (batch) | $val_loss (validation)"
		push!(history, :training_loss, i, l)
		push!(history, :validation_likelihood, i, val_loss)
		
		if val_loss < best_val_loss
			best_val_loss = val_loss
			patience = p.patience

			# this should save the model at least once
			# when the validation loss is decreasing 
			if mod(i, 10) == 0
				model = deepcopy(trn_model)
			end
		else
			patience -= 1
			if patience == 0
				@info "Stopped training at $(i) iteration."
				break
			end
		end
		i += 1
	end

	# returning model in this way is not ideal
	# it would have to modify the reference the
	# underlying structure
	(history=history, iterations=i, model=model)
end

function StatsBase.predict(model::RealNVPFlow, X)
	Z, logJ = model((X, _init_logJ(X)))
    -(logpdf(model.base, Z)' .+ logJ)
end
