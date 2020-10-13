"""
	function build_mlp(idim::Int, hdim::Int, odim::Int, nlayers::Int; activation::String="relu", lastlayer::String="")
	Creates a chain with `nlayers` layers of `hdim` neurons with transfer function `activation`.
	input and output dimension is `idim` / `odim`
	If lastlayer is no specified, all layers use the same function.
	If lastlayer is "linear", then the last layer is forced to be Dense.
	It is also possible to specify dimensions in a vector.

```juliadoctest
julia> build_mlp(4, 11, 1, 3, activation="relu")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, relu))

julia> build_mlp([4, 11, 11, 1], activation="relu")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, relu))

julia> build_mlp(4, 11, 1, 3, activation="relu", lastlayer="tanh")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, tanh))
```
"""
build_mlp(ks::Vector{Int}, fs::Vector) = Flux.Chain(map(i -> Dense(i[2],i[3],i[1]), zip(fs,ks[1:end-1],ks[2:end]))...)

build_mlp(idim::Int, hdim::Int, odim::Int, nlayers::Int; kwargs...) =
	build_mlp(vcat(idim, fill(hdim, nlayers-1)..., odim); kwargs...)

function build_mlp(ks::Vector{Int}; activation::String = "relu", lastlayer::String = "")
	activation = (activation == "linear") ? "identity" : activation
	fs = Array{Any}(fill(eval(:($(Symbol(activation)))), length(ks) - 1))
	if !isempty(lastlayer)
		fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
	end
	build_mlp(ks, fs)
end

"""
	vec_size(conv_dims, scalings, channel)

Computes the size of vectorized ouput, given size of input, convolutional scalings and last channel.
"""
function vec_size(conv_dims, scalings, channel)
	ho = conv_dims[1]/(reduce(*, scalings)) # height before reshaping
	wo = conv_dims[2]/(reduce(*, scalings)) # width before reshaping
	# this ensures size compatibility, ho/wo are Ints
	(ho == floor(Int, ho)) ? ho = floor(Int, ho) : error("your input size and scaling is not compatible")
	(wo == floor(Int, wo)) ? wo = floor(Int, wo) : error("your input size and scaling is not compatible")
	ho*wo*channel, ho, wo
end

# these two functions are taken from GMExtensions.jl, but I wanted to avoid the dependency since
# it is an unregistered package
# TODO: maybe use this in GANomaly constructor?
"""
	conv_encoder(idim, zdim, kernelsizes, channels, scalings[; activation, densedims])

Constructs a convolutional encoder.

# Arguments
- `idim`: size of input - (h,w,c)
- `zdim`: latent space dimension
- `kernelsizes`: kernelsizes for conv. layers (only odd numbers are allowed)
- `channels`: channel numbers
- `scalings`: scalings vector
- `activation`: default relu
- `densedims`: if set, more than one dense layers are used 

# Example
```julia-repl
julia> encoder = conv_encoder((64, 48, 1), 2, (3, 5, 5), (2, 4, 8), (2, 2, 2), densedims = (256,))
Chain(Conv((3, 3), 1=>2, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), Conv((5, 5), 2=>4, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), Conv((5, 5), 4=>8, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), #44, Dense(384, 2))
julia> encoder(randn(Float32, 64, 48, 1, 2))
2×2 Array{Float32,2}:
  0.247844   0.133781
 -0.605763  -0.494911
```
"""
function conv_encoder(idim::Union{Tuple, Vector}, zdim::Int, kernelsizes::Union{Tuple, Vector}, 
	channels::Union{Tuple, Vector}, scalings::Union{Tuple, Vector}; 
	activation = "relu", densedims::Union{Tuple, Vector} = [], batchnorm=false)
	nconv = length(kernelsizes)
	(nconv == length(channels) == length(scalings)) ? nothing : error("incompatible input dimensions")
	(length(idim) == 3) ? nothing : error("idim must be (h, w, c)")
	# also check that kernelsizes are all odd numbers
	(all(kernelsizes .% 2 .== 1)) ? nothing : error("not implemented for odd kernelsizes")

	# string to function
	actf = eval(:($(Symbol(activation))))
	
	# initialize dimensions
	cins = vcat(idim[3], channels[1:end-1]...) # channels in
	couts = channels # channels out
	din, _, _ = vec_size(idim, scalings, channels[end]) # the size of vectorized output after all convolutions

	# now build a vector of layers to be used later
	layers = Array{Any,1}()
	# first add the convolutional and maxpooling layers
	for (k, ci, co, s) in zip(kernelsizes, cins, couts, scalings)
		pad = Int((k-1)/2)
		# paddding so that input and output size are same
		push!(layers, Conv((k,k), ci=>co, actf; pad = (pad, pad))) 
		push!(layers, MaxPool((s,s)))
		# generally, bn is not recommended for training vaes
		batchnorm ? push!(layers, BatchNorm(co)) : nothing
	end

	# reshape
	push!(layers, x -> reshape(x, din, :))

	# and dense layers
	ndense = length(densedims)
	dins = vcat(din, densedims...)
	douts = vcat(densedims..., zdim)
	dacts = vcat([actf for _ in 1:ndense]..., identity)
	for (_di, _do, _da) in zip(dins, douts, dacts)
		push!(layers, Dense(_di, _do, _da))
	end

	Flux.Chain(layers...)
end

"""
	conv_decoder(odim, zdim, kernelsizes, channels, scalings[; activation, densedims, vec_output,
		vec_output_dim])

Constructs a convolutional encoder.

# Arguments
- `odim`: size of outnput - (h,w,c)
- `zdim`: latent space dimension
- `kernelsizes`: kernelsizes for conv. layers (only odd numbers are allowed)
- `channels`: channel numbers
- `scalings`: scalings vector
- `activation`: default relu
- `densedims`: if set, more than one dense layers are used 
- `vec_output`: output is vectorized (default false)
- `vec_output_dim`: determine what the final size of the vectorized input should be (e.g. add extra dimensions for variance estimation)

# Example
```julia-repl
julia> decoder = conv_decoder((64, 48, 1), 2, (5, 5, 3), (8, 4, 2), (2, 2, 2); densedims = (256,))
Chain(Dense(2, 256, relu), Dense(256, 384, relu), #19, ConvTranspose((2, 2), 8=>8, relu), Conv((5, 5), 8=>4, relu), ConvTranspose((2, 2), 4=>4, relu), Conv((5, 5), 4=>2, relu), ConvTranspose((2, 2), 2=>2, relu), Conv((3, 3), 2=>1))
julia> y = decoder(randn(Float32, 2, 2));
julia> size(y)
(64, 48, 1, 2)
```
"""
function conv_decoder(odim::Union{Tuple, Vector}, zdim::Int, kernelsizes::Union{Tuple, Vector}, 
	channels::Union{Tuple, Vector}, scalings::Union{Tuple, Vector}; 
	activation = relu, densedims::Union{Tuple, Vector} = [], 
	vec_output = false, vec_output_dim = nothing, batchnorm=false)
	nconv = length(kernelsizes)
	(nconv == length(channels) == length(scalings)) ? nothing : error("incompatible input dimensions")
	(length(odim) == 3) ? nothing : error("odim must be (h, w, c)")
	# also check that kernelsizes are all odd numbers
	(all(kernelsizes .% 2 .== 1)) ? nothing : error("not implemented for odd kernelsizes")

	# string to function
	actf = eval(:($(Symbol(activation))))
	
	# initialize some stuff
	cins = channels # channels in
	couts = vcat(channels[2:end]..., odim[3]) # channels out
	dout, ho, wo = vec_size(odim, scalings, channels[1]) # size of vectorized output after convolutions

	# now build a vector of layers to be used later
	layers = Array{Any,1}()

	# start with dense layers
	ndense = length(densedims)
	dins = vcat(zdim, densedims...)
	douts = vcat(densedims..., dout)
	for (_di, _do) in zip(dins, douts)
		push!(layers, Dense(_di, _do, actf))
	end

	# reshape
	push!(layers, x -> reshape(x, ho, wo, channels[1], :))

	# add the transpose and convolutional layers
	acts = vcat([actf for _ in 1:nconv-1]..., identity) 
	for (k, ci, co, s, act) in zip(kernelsizes, cins, couts, scalings, acts)
		pad = Int((k-1)/2)
		# use convtranspose for upscaling - there are other posibilities, however this seems to be ok
		push!(layers, ConvTranspose((s,s), ci=>ci, actf, stride = (s,s))) 
		push!(layers, Conv((k,k), ci=>co, act; pad = (pad, pad)))
		# generally, bn is not recommended for training vaes
		batchnorm ? push!(layers, BatchNorm(co)) : nothing
	end

	# if you want vectorized output
	if vec_output
		push!(layers, x -> reshape(x, reduce(*, odim), :))
		# if you want the final vector to be a different size than the output 
		# (e.g. you want extra dimensions for variance estimation)
		(vec_output_dim == nothing) ? nothing : push!(layers, Dense(reduce(*, odim), vec_output_dim))
	end
	
	Flux.Chain(layers...)
end