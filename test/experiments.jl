@testset "Experiments" begin
	using DrWatson
	@quickactivate
	using Test
	using GenerativeAD

	# first create dummy data
	tstpath = abspath(joinpath(dirname(pathof(GenerativeAD)), "../test/misc"))
	run(`mkdir $tstpath`)
	run(`touch $tstpath/model_a=4_b=0.001.bson`)
	run(`touch $tstpath/a=4_b=0.001.bson`)
	run(`touch $tstpath/a=4_b=0.001_c=bar.bson`)
	run(`touch $tstpath/a=4_b=0.001_d=foo.bson`)
	
	# now test check_params
	@test !GenerativeAD.check_params(tstpath, (a=4,))
	@test !GenerativeAD.check_params(tstpath, (a=4, b=0.001))
	@test GenerativeAD.check_params(tstpath, (a=4, b=0.002))
	@test !GenerativeAD.check_params(tstpath, (a=4, b=0.001, c="bar"))
	@test GenerativeAD.check_params(tstpath, (a=4, b=0.001, c="foo"))
	@test !GenerativeAD.check_params(tstpath, (a=4, b=0.001, d="foo"))
	@test GenerativeAD.check_params(tstpath, (a=4, b=0.001, g="a"))

	# cleanup
	run(`rm -r $tstpath`)
end