using GenerativeAD
using FileIO, BSON
using ValueHistories, DistributionsAD
using Flux
using ConditionalDists
using GenerativeModels
using EvalMetrics
using Plots

f = "vae/iris/seed=3/model_activation=tanh_batchsize=32_hdim=256_init_seed=59386742_lr=0.001_nlayers=4_zdim=2.bson"
#######
function get_crit_vae(f)
	model_data = load(f)
	loss = get(model_data["history"], :validation_likelihood)[2][end]
end
#######

#path = "vae/iris/seed=3"
path = "vae/htru2/seed=3"
mfs = filter(x->occursin("model", x), readdir(path, join=true))

crit_vals = map(get_crit_vae, mfs)

seeds = map(x->split(split(x, "init_seed")[2], "_")[1][2:end], mfs)
sfs = filter(x->occursin("reconstruction-sampled", x), readdir(path, join=true))


function get_auc(sfs, seed)
	f = sfs[findfirst(x->occursin(seed,x), sfs)]
	data = load(f)
	auc = EvalMetrics.auc_trapezoidal(EvalMetrics.roccurve(data[:val_labels], vec(data[:val_scores]))...)
end

aucs = map(x->get_auc(sfs, x), seeds)

scatter(crit_vals, aucs, xlabel="validation ELBO", ylabel="validation AUC", title="VAE - iris")
savefig("~/generativead/GenerativeAD.jl/plots/elbo_vs_auc.png")

function get_auc_test(sfs, seed)
	f = sfs[findfirst(x->occursin(seed,x), sfs)]
	data = load(f)
	auc = EvalMetrics.auc_trapezoidal(EvalMetrics.roccurve(data[:tst_labels], vec(data[:tst_scores]))...)
end
aucs = map(x->get_auc_test(sfs, x), seeds)
scatter(crit_vals, aucs, xlabel="validation ELBO", ylabel="test AUC", title="VAE - iris")
savefig("~/generativead/GenerativeAD.jl/plots/elbo_vs_auc_test.png")



sfs = filter(x->occursin("reconstruction-sampled", x), readdir(path, join=true))

function return_rec_auc(f)
	data = load(f)
	auc = EvalMetrics.auc_trapezoidal(EvalMetrics.roccurve(data[:val_labels], vec(data[:val_scores]))...)
	rec = Flux.mean(data[:val_scores][data[:val_labels].==0])
	auc, rec
end

res = map(return_rec_auc, sfs)
aucs = [x[1] for x in res]
recs = [x[2] for x in res]
scatter(recs, aucs, xlabel="validation reconstruction error", ylabel="validation AUC", title="VAE - iris")
savefig("~/generativead/GenerativeAD.jl/plots/rec_sampled_vs_auc.png")

## WAE
############
f = "wae/iris/seed=3/model_activation=tanh_batchsize=64_hdim=512_init_seed=86274140_kernel=gauss_lambda=0.1_lr=0.0001_nlayers=4_sigma=0.001_zdim=2.bson"
model_data = load(f)

#############
function get_crit_wae(f)
	model_data = load(f)
	loss = get(model_data["history"], :validation_likelihood)[2][end]
end
function get_crit_aae(f)
	model_data = load(f)
	loss = get(model_data["history"], :validation_loss)[2][end]
end
#############

path = "wae/iris/seed=3"
mfs = filter(x->occursin("model", x), readdir(path, join=true))
crit_vals = map(get_crit_wae, mfs)
aucs = map(x->get_auc(sfs, x), seeds)
scatter(crit_vals, aucs, xlabel="validation loss", ylabel="validation AUC", title="WAE - iris")

seeds = map(x->split(split(x, "init_seed")[2], "_")[1][2:end], mfs)
sfs = filter(x->occursin("reconstruction-sampled", x), readdir(path, join=true))
res = map(return_rec_auc, sfs)
aucs = [x[1] for x in res]
recs = [x[2] for x in res]
scatter(recs, aucs, xlabel="validation reconstruction error", ylabel="validation AUC", title="WAE - iris")

path = "aae/iris/seed=3"
mfs = filter(x->occursin("model", x), readdir(path, join=true))
crit_vals = map(get_crit_aae, mfs)
aucs = map(x->get_auc(sfs, x), seeds)
scatter(crit_vals, aucs, xlabel="validation loss", ylabel="validation AUC", title="AAE - iris")

seeds = map(x->split(split(x, "init_seed")[2], "_")[1][2:end], mfs)
sfs = filter(x->occursin("reconstruction-sampled", x), readdir(path, join=true))
res = map(return_rec_auc, sfs)
aucs = [x[1] for x in res]
recs = [x[2] for x in res]
scatter(recs, aucs, xlabel="validation reconstruction error", ylabel="validation AUC", title="AAE - iris")

#### FMGAN