using DrWatson
@quickactivate
using Test
using GenerativeAD

@testset "Models" begin
    include("knn.jl")
    include("vae.jl")
    include("aae.jl")
    include("gan.jl")
end