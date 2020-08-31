@testset "KNN" begin
	using Test
	import GMAD.Models.KNNAnomaly
	import StatsBase: fit!, predict

	M = 4
	N1 = 100
	N2 = 50
	X = randn(M, N1)
	Y = randn(M, N2)
	model = KNNAnomaly(3, :gamma)
	@test_throws ErrorException predict(model, Y)
	fit!(model, X)
	@test length(predict(model, Y)) == N2
	@test all(predict(model, Y, 5) .!= predict(model, Y))
	@test length(predict(model, Y, 5)) == N2
	@test all(predict(model, Y, 3, :kappa) .!= predict(model, Y))
	@test length(predict(model, Y, 3, :kappa)) == N2
end