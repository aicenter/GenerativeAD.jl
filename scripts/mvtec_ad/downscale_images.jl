using DrWatson
@quickactivate
using GenerativeAD
using BSON, FileIO

factor = 4

# downscale from 1024x1024 to 256x256
function downscale(x::AbstractArray{T,4}, factor) where T
	h,w,c,n = size(x)
	(w == h == 1024) ? nothing : error("only implemented for downscaling images from 1024 pixels")
	nw = nh = Int(h/factor)
	y = similar(x, nh, nw, c, n)
	step = factor-1
	for ni in 1:n
		for ic in 1:c
			for iw in 1:nw
				for ih in 1:nh
					y[ih, iw, ic, ni] = sum(x[(factor*ih-step):(factor*ih), (factor*iw-step):(factor*iw), ic, ni])/factor^2
				end
			end
		end
	end
	y
end

outpath = datadir("mvtec_ad/downscaled_data")
mkpath(outpath)

# start with wood
function downscale_data(category, factor)
	normal, anomalous = GenerativeAD.Datasets.load_mvtec_ad_data(category=category)
	dnormal, danomalous = map(x->downscale(x,factor), (normal, anomalous))
	s = Int(1024/factor)
	fname = joinpath(outpath, "$(category)_$(s).bson")
	save(fname, Dict(:normal => dnormal, :anomalous => danomalous))
	@info "saved to $fname"
end

map(f->map(c->downscale_data(c,f), ["wood", "transistor", "grid", "bottle", "metal_nut", "pill", 
	"capsule"]), [4, 8, 16, 32])