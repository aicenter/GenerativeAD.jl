@testset "DATA" begin
    dataset = "abalone"
	data, _, _ = UCI.get_loda_data(dataset)
	cdata = UCI.normalize(hcat(data.normal, data.easy, data.medium))  # this is a control array
	na = size(data.easy,2) + size(data.medium,2)

	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y) = GMAD.load_data(dataset)	
	@test size(tr_x,1) == size(data.normal,1)
	@test size(tr_x,2) == floor(Int,size(data.normal,2)*0.6) == length(tr_y)
	@test sum(tr_y) == 0
	@test size(val_x,2) == floor(Int,size(data.normal,2)*0.2) + floor(Int,na*0.5) == length(val_y) > sum(val_y) > 0
	@test size(tst_x,2) == floor(Int,size(data.normal,2)*0.2) + floor(Int,na*0.5) == length(tst_y) > sum(tst_y) > 0
end