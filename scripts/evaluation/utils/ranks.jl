# ideally this should be in the GenerativeAD pkg
function compute_ranks(rt)
	mask_nan_max = (x) -> (isnan(x) ? -Inf : x)
	rs = []
	for row in eachrow(rt)
		push!(rs, StatsBase.competerank(mask_nan_max.(Vector(row)), rev = true))
	end
	# each row represents ranks of a method
	reduce(hcat, rs)
end
