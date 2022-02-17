using DrWatson
@quickactivate
using GenerativeAD
using BSON, FileIO
using ImageTransformations

outpath = datadir("mvtec_ad/downscaled_data")
mkpath(outpath)

# start with wood
function downscale_data(category, size)
	normal, anomalous = GenerativeAD.Datasets.load_mvtec_ad_data(category=category)
	dnormal, danomalous = map(x->imresize(x,(size,size)), (normal, anomalous))
	fname = joinpath(outpath, "$(category)_$(size).bson")
	save(fname, Dict(:normal => dnormal, :anomalous => danomalous))
	@info "saved to $fname"
end

map(f->map(c->downscale_data(c,f), ["wood", "transistor", "grid"]), [256, 128, 64, 32])
