"""
	function build_mlp(isize::Int, hsize::Int, osize::Int, nlayers::Int; ftype::String="relu", lastlayer::String="")
	Creates a chain with `nlayers` layers of `hsize` neurons with transfer function `ftype`.
	input and output dimension is `isize` / `osize`
	If lastlayer is no specified, all layers use the same function.
	If lastlayer is "linear", then the last layer is forced to be Dense.
	It is also possible to specify dimensions in a vector.

```juliadoctest
julia> build_mlp(4, 11, 1, 3, ftype="relu")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, relu))

julia> build_mlp([4, 11, 11, 1], ftype="relu")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, relu))

julia> build_mlp(4, 11, 1, 3, ftype="relu", lastlayer="tanh")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, tanh))
```
"""
build_mlp(ks::Vector{Int}, fs::Vector) = Flux.Chain(map(i -> Dense(i[2],i[3],i[1]), zip(fs,ks[1:end-1],ks[2:end]))...)

build_mlp(isize::Int, hsize::Int, osize::Int, nlayers::Int; kwargs...) =
	build_mlp(vcat(isize, fill(hsize, nlayers-1)..., osize); kwargs...)

function build_mlp(ks::Vector{Int}; ftype::String = "relu", lastlayer::String = "")
	ftype = (ftype == "linear") ? "identity" : ftype
	fs = Array{Any}(fill(eval(:($(Symbol(ftype)))), length(ks) - 1))
	if !isempty(lastlayer)
		fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
	end
	build_mlp(ks, fs)
end