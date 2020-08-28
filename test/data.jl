@testset "DATA" begin
    dataset = "abalone"
	data, _, _ = UCI.get_loda_data(dataset)
	data = UCI.normalize(data) 

end