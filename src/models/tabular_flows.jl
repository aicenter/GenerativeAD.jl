using Flux
using StatsBase
using Distributions
using DistributionsAD
using ContinuousFlows: RealNVP, MaskedAutoregressiveFlow
using ValueHistories
using MLDataPattern: RandomBatches

abstract type TabularFlow end

struct RealNVPFlow <: TabularFlow
	flows
	base
end

function RealNVPFlow(nflows::Int, idim::Int, hdim::Int, nlayers::Int)
	RealNVPFlow(Chain([
		RealNVP(
			idim, 
			(d, o, act, postprocess) -> build_mlp(d, hdim, o, nlayers, activation=act, lastlayer=postprocess),
			mod(i,2) == 0)
		for i in 1:nflows]...), MvNormal(idim, 1.0f0))
end

(nvpf::RealNVPFlow)(X) = nvpf.flows(X)
Flux.trainable(nvpf::RealNVPFlow) = (nvpf.flows, )

struct MAF <: TabularFlow
	flows
	base
end

function MAF(nflows::Int, idim::Int, hdim::Int, nlayers::Int, ordering::String)
	MAF(Chain([
        MaskedAutoregressiveFlow(
            idim, 
            hdim,
            nlayers, 
            idim, 
            (ordering == "natural") ? (
                (mod(i, 2) == 0) ? "reversed" : "sequential"
              ) : "random"
            ) 
        for i in 1:nflows]...), MvNormal(idim, 1.0f0))
end

(maf::MAF)(X) = maf.flows(X)
Flux.trainable(maf::MAF) = (maf.flows, )


function loss(model::F, X) where {F <: TabularFlow}
    Z, logJ = model((X, _init_logJ(X)))
    -sum(logpdf(model.base, Z)' .+ logJ)/size(X, 2)
end

function StatsBase.fit!(model::F, data::Tuple, p) where F <: TabularFlow
	opt = Flux.ADAM(p.lr)
	
	trn_model = deepcopy(model)
	ps = Flux.params(trn_model);

	X = data[1][1]
	# filter only normal data from validation set
	X_val = data[2][1][:, data[2][2] .== 0.0f0]

	history = MVHistory()
	patience = p.patience
	reg = (p.wreg > 0) ? l2_reg : _ -> 0.0
	
	best_val_loss = loss(trn_model, X_val)
	i = 1
	start_time = time()
	for batch in RandomBatches(X, p.batchsize)
		l = 0.0f0
		gs = gradient(() -> begin l = loss(trn_model, batch) + p.wreg*reg(ps) end, ps)
		Flux.update!(opt, ps, gs)

		# validation/early stopping
		val_loss = loss(trn_model, X_val)
		@info "$i - loss: $l (batch) | $val_loss (validation)"
		
		if isnan(val_loss) || isnan(l)
			error("Encountered invalid values in loss function.")
		end

		push!(history, :training_loss, i, l)
		push!(history, :validation_likelihood, i, val_loss)
		
		# 23 hours time limit
		if time() - start_time > 82600
			@info "Stopped training after $(i) iterations due to time limit."
			model = deepcopy(trn_model)
			break
		end

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
				@info "Stopped training after $(i) iterations."
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

function StatsBase.predict(model::F, X) where {F <: TabularFlow}
	Z, logJ = model((X, _init_logJ(X)))
    -(logpdf(model.base, Z)' .+ logJ)
end
