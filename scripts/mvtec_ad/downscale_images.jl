using DrWatson
@quickactivate
using GenerativeAD
using BSON, FileIO

# downscale from 1024x1024 to 256x256
function downscale(x::AbstractArray{T,4}) where T
	h,w,c,n = size(x)
	(w == h == 1024) ? nothing : error("only implemented for downscaling images from 1024 to 256 pixels")
	y = similar(x, 256, 256, c, n)
	for ni in 1:n
		for ic in 1:c
			for iw in 1:256
				for ih in 1:256
					y[ih, iw, ic, ni] = sum(x[(4*ih-3):(4*ih), (4*iw-3):(4*iw), ic, ni])/16
				end
			end
		end
	end
	y
end

outpath = datadir("mvtec_ad/downscaled_data")
mkpath(outpath)

# start with wood
function downscale_data(category)
	normal, anomalous = GenerativeAD.Datasets.load_mvtec_ad_data(category=category)
	dnormal, danomalous = map(downscale, (normal, anomalous))
	fname = joinpath(outpath, "$(category).bson")
	save(fname, Dict(:normal => dnormal, :anomalous => danomalous))
	@info "saved to $fname"
end

map(downscale_data, ["wood", "transistor", "grid"])
