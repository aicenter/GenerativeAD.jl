# this contains stuff that is common for VAE, WAE and AAE models
"""
	AEModel

A Union of VAE and AAE types.
"""
const AEModel = Union{VAE, AAE}

# these functions need to be overloaded for convolutional models
function ConditionalDists.condition(p::ConditionalMvNormal, z::AbstractArray{T,4}) where T
	(μ,σ) = ConditionalDists.mean_var(p.mapping(z))
	if size(σ,1) == 1
		σ = dropdims(σ, dims=1)
	end
	ConditionalDists.BatchMvNormal(μ,σ)
end
Distributions.logpdf(d::ConditionalDists.BMN, x::AbstractArray{T,4}) where T<:Real =
	Distributions.logpdf(d, reshape(x,:,size(x,4)))
Distributions.mean(p::ConditionalMvNormal, z::AbstractArray{T,4}) where T<:Real = mean(condition(p,z))
Distributions.var(p::ConditionalMvNormal, z::AbstractArray{T,4}) where T<:Real = var(condition(p,z))
Distributions.rand(p::ConditionalMvNormal, z::AbstractArray{T,4}) where T<:Real = rand(condition(p,z))
Distributions.logpdf(p::ConditionalMvNormal, x::AbstractArray{T,4}, z::AbstractArray{T,2}) where T<:Real = 
	logpdf(condition(p,z), x)

"""
	reconstruct(model::AEModel, x)

Data reconstruction.
"""
reconstruct(model::AEModel, x) = mean(model.decoder, rand(model.encoder, x))
reconstruct(model::AEModel, x::AbstractArray{T,4}) where T = 
	reshape(mean(model.decoder, rand(model.encoder, x)), size(x)...)

"""
	generate(model::AEModel, N::Int[, outdim])

Data generation. Support output dimension if the output needs to be reshaped, e.g. in convnets.
"""
generate(model::AEModel, N::Int) = mean(model.decoder, rand(model.prior, N))
generate(model::AEModel, N::Int, outdim) = reshape(generate(model, N), outdim..., :)

"""
	encode_mean(model::AEModel, x)

Produce data encodings.
"""
encode_mean(model::AEModel, x) = mean(model.encoder, x)
"""
	encode_mean_gpu(model::AEModel, x[, batchsize])

Produce data encodings. Works only on 4D tensors.
"""
encode_mean_gpu(model, x::AbstractArray{T,4}) where T = encode_mean(model, gpu(Array(x)))
function encode_mean_gpu(model, x::AbstractArray{T,4}, batchsize::Int) where T
	# this has to be done in a loop since doing cat(map()...) fails if there are too many arguments
	dl = Flux.Data.DataLoader(x, batchsize=batchsize)
	z = encode_mean_gpu(model, iterate(dl,1)[1])
	N = size(x, ndims(x))
	encodings = gpu(zeros(eltype(z), size(z,1), N))
	for (i,batch) in enumerate(dl)
		encodings[:,1+(i-1)*batchsize:min(i*batchsize,N)] .= encode_mean_gpu(model, batch)
	end
	encodings
end

"""
	reconstruction_score(model::AEModel, x)

Anomaly score based on the reconstruction probability of the data.
"""
function reconstruction_score(model::AEModel, x) 
	p = condition(model.decoder, rand(model.encoder, x))
	-logpdf(p, x)
end
"""
	reconstruction_score_mean(model::AEModel, x)

Anomaly score based on the reconstruction probability of the data. Uses mean of encoding.
"""
function reconstruction_score_mean(model::AEModel, x) 
	p = condition(model.decoder, mean(model.encoder, x))
	-logpdf(p, x)
end
"""
	latent_score(model::AEModel, x) 

Anomaly score based on the similarity of the encoded data and the prior.
"""
function latent_score(model::AEModel, x) 
	z = rand(model.encoder, x)
	-logpdf(model.prior, z)
end

"""
	latent_score_mean(model::AEModel, x) 

Anomaly score based on the similarity of the encoded data and the prior. Uses mean of encoding.
"""
function latent_score_mean(model::AEModel, x) 
	z = mean(model.encoder, x)
	-logpdf(model.prior, z)
end
