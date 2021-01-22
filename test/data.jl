@testset "DATA" begin
	# UCI datasets
    dataset = "abalone"
	data, _, _ = UCI.get_loda_data(dataset)
	cdata = UCI.normalize(hcat(data.normal, data.easy, data.medium))  # this is a control array
	na = size(data.easy,2) + size(data.medium,2)

	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y) = GenerativeAD.load_data(dataset)	
	@test size(tr_x,1) == size(data.normal,1)
	@test size(tr_x,2) == floor(Int,size(data.normal,2)*0.6) == length(tr_y)
	@test sum(tr_y) == 0
	@test size(val_x,2) == floor(Int,size(data.normal,2)*0.2) + floor(Int,na*0.5) == length(val_y) > sum(val_y) > 0
	@test size(tst_x,2) == floor(Int,size(data.normal,2)*0.2) + floor(Int,na*0.5) == length(tst_y) > sum(tst_y) > 0

	# test contamination
	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y) = GenerativeAD.load_data("waveform-1", contamination=0.1)
	@test abs(sum(tr_y)/length(tr_y) - 0.1) < 0.001
	@test sum(val_y) == sum(tst_y)

	# img datasets
	function test_img_data(dataset::String, dims, contamination, method="leave-one-out")
		(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y) = GenerativeAD.load_data(dataset, contamination=contamination, method=method)
		h,w,c,n = dims
		@test size(tr_x)[1:3] == size(val_x)[1:3] == size(tst_x)[1:3] == (h,w,c)
		@test n-3 <= size(tr_x,4) + size(val_x,4) + size(tst_x,4) <= n # some samples are left out due to rounding
		@test abs(sum(tr_y)/length(tr_y) - contamination) < 0.001
		if method == "leave-one-in"
			@test size(tr_x,4) < size(val_x,4)
		end
	end

	test_img_data("MNIST", (28, 28, 1, 70000), 0.0)
	test_img_data("FashionMNIST", (28, 28, 1, 70000), 0.0)
	test_img_data("MNIST", (28, 28, 1, 70000), 0.05)
	test_img_data("FashionMNIST", (28, 28, 1, 70000), 0.05)
	test_img_data("MNIST", (28, 28, 1, 70000), 0.2, "leave-one-in")
	test_img_data("FashionMNIST", (28, 28, 1, 70000), 0.2, "leave-one-in")

	# this takes a lot of time...
	if !complete
		@info "Skipping tests on CIFAR10 and SVHN2..."
	else
		@info "Testing CIFA10 and SVHN2 datasets"
		test_img_data("CIFAR10", (32, 32, 3, 60000), 0.0)
		test_img_data("SVHN2", (32, 32, 3, 99289), 0.0)
	end
end