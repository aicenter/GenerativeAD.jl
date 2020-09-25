@testset "Experiments" begin
	using DrWatson
	@quickactivate
	using Test
	using GenerativeAD

	tstpath = joinpath(dirname(pathof(GenerativeAD)), "../test/misc")
	@test !GenerativeAD.check_params(tstpath, (a=4,))
	@test !GenerativeAD.check_params(tstpath, (a=4, b=0.001))
	@test GenerativeAD.check_params(tstpath, (a=4, b=0.002))
	@test !GenerativeAD.check_params(tstpath, (a=4, b=0.001, c="bar"))
	@test GenerativeAD.check_params(tstpath, (a=4, b=0.001, c="foo"))
	@test !GenerativeAD.check_params(tstpath, (a=4, b=0.001, d="foo"))
	@test GenerativeAD.check_params(tstpath, (a=4, b=0.001, g="a"))
end