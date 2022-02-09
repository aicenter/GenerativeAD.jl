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

	# wildlife MNIST
	nci = 2
	(xn, yn), (xa, ya)=GenerativeAD.Datasets.load_wildlife_mnist_data(normal_class_ind=nci, denormalize=true)
	@test all(yn .== nci)
	@test ndims(yn) == 1
	@test length(yn) == size(xn, 4)
	@test all(ya .!= nci)
	@test length(unique(ya)) == 9
	@test ndims(ya) == 1
	@test length(ya) == size(xa, 4)

	nci = 2
	aci = 3
	(xn, yn), (xa, ya)=GenerativeAD.Datasets.load_wildlife_mnist_data(normal_class_ind=nci, 
		anomaly_class_ind = aci, denormalize=true)
	@test all(yn .== nci)
	@test ndims(yn) == 1
	@test length(yn) == size(xn, 4)
	@test all(ya .== aci)
	@test length(unique(ya)) == 1
	@test ndims(ya) == 1
	@test length(ya) == size(xa, 4)

	nci = [2, 4, 8]
	(xn, yn), (xa, ya)=GenerativeAD.Datasets.load_wildlife_mnist_data(normal_class_ind=nci, denormalize=true)
	@test all(map(x->x in nci, yn))
	@test length(unique(yn)) == length(nci)
	@test ndims(yn) == 1
	@test length(yn) == size(xn, 4)
	@test all(map(x -> !(x in nci), ya))
	@test length(unique(ya)) == 10 - length(nci)
	@test ndims(ya) == 1
	@test length(ya) == size(xa, 4)

	nci = [2, 4, 8]
	aci = [1, 5, 9]
	(xn, yn), (xa, ya)=GenerativeAD.Datasets.load_wildlife_mnist_data(normal_class_ind=nci, 
		anomaly_class_ind = aci, denormalize=true)
	@test all(map(x->x in nci, yn))
	@test length(unique(yn)) == length(nci)
	@test ndims(yn) == 1
	@test length(yn) == size(xn, 4)
	@test all(map(x -> (x in aci), ya))
	@test length(unique(ya)) == length(aci)
	@test ndims(ya) == 1
	@test length(ya) == size(xa, 4)

	nci = (1, -1, -1)
	(xn, yn), (xa, ya)=GenerativeAD.Datasets.load_wildlife_mnist_data(normal_class_ind=nci, denormalize=true)
	@test all(yn[1,:] .== nci[1]) 
	@test length(unique(yn[2,:])) == 10
	@test length(unique(yn[3,:])) == 10
	@test size(yn,2) == size(xn, 4)
	@test all(ya[1,:] .!= nci[1]) 
	@test length(unique(ya[1,:])) == 9
	@test length(unique(ya[2,:])) == 10
	@test length(unique(ya[3,:])) == 10
	@test size(ya,2) == size(xa, 4)

	nci = (1, 1, -1)
	(xn, yn), (xa, ya)=GenerativeAD.Datasets.load_wildlife_mnist_data(normal_class_ind=nci, denormalize=true)
	@test all(yn[1,:] .== nci[1]) 
	@test all(yn[2,:] .== nci[2]) 
	@test length(unique(yn[3,:])) == 10
	@test size(yn,2) == size(xn, 4)
	@test all((ya[1,:] .!= nci[1]) .| (ya[2,:] .!= nci[2]))
	@test length(unique(ya[1,:])) == 10
	@test length(unique(ya[2,:])) == 10
	@test length(unique(ya[3,:])) == 10
	@test size(ya,2) == size(xa, 4)

	nci = (1, 1, 1)
	(xn, yn), (xa, ya)=GenerativeAD.Datasets.load_wildlife_mnist_data(normal_class_ind=nci, denormalize=true)
	@test all(yn[1,:] .== nci[1]) 
	@test all(yn[2,:] .== nci[2]) 
	@test all(yn[3,:] .== nci[3]) 
	@test size(yn,2) == size(xn, 4)
	@test all((ya[1,:] .!= nci[1]) .| (ya[2,:] .!= nci[2]) .| (ya[3,:] .!= nci[3]))
	@test length(unique(ya[1,:])) == 10
	@test length(unique(ya[2,:])) == 10
	@test length(unique(ya[3,:])) == 10
	@test size(ya,2) == size(xa, 4)

	nci = (1, 1, 1)
	aci = (2, 3, 5)
	(xn, yn), (xa, ya)=GenerativeAD.Datasets.load_wildlife_mnist_data(normal_class_ind=nci, 
		anomaly_class_ind=aci, denormalize=true)
	@test all(yn[1,:] .== nci[1]) 
	@test all(yn[2,:] .== nci[2]) 
	@test all(yn[3,:] .== nci[3]) 
	@test size(yn,2) == size(xn, 4)
	@test all(ya[1,:] .== aci[1]) 
	@test all(ya[2,:] .== aci[2]) 
	@test all(ya[3,:] .== aci[3]) 
	@test size(ya,2) == size(xa, 4)

	nci = 1
	aci = (2, 3, 5)
	(xn, yn), (xa, ya)=GenerativeAD.Datasets.load_wildlife_mnist_data(normal_class_ind=nci, 
		anomaly_class_ind=aci, denormalize=true)
	@test all(yn .== nci) 
	@test length(yn) == size(xn, 4)
	@test all(ya[1,:] .== aci[1]) 
	@test all(ya[2,:] .== aci[2]) 
	@test all(ya[3,:] .== aci[3]) 
	@test size(ya,2) == size(xa, 4)

	# this takes a lot of time...
	if !complete
		@info "Skipping tests on CIFAR10 and SVHN2..."
	else
		@info "Testing CIFA10 and SVHN2 datasets"
		test_img_data("CIFAR10", (32, 32, 3, 60000), 0.0)
		test_img_data("SVHN2", (32, 32, 3, 99289), 0.0)
	end
end