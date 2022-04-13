using DrWatson
@quickactivate
using BSON, FileIO
using NPZ

dataset = "CIFAR10"
ac = 7
seed = 1
modelname = "sgvae"

indir = datadir("sgad_latent_scores/images_leave-one-in/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
outdir = datadir("sgad_latent_scores/images_leave-one-in_converted/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
mkpath(outdir)

function convert_save(f,indir,outdir)
	data = load(joinpath(indir, f))
	_f = split(f, ".bson")[1]
	for (s, l) in zip(["train", "validation", "test"], [:tr_scores, :val_scores, :tst_scores])
		outf = joinpath(outdir, _f * "_$(s).npy")
		npzwrite(outf, data[l])
	end
end

for f in readdir(indir)
	convert_save(f,indir,outdir)
end
