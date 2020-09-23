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