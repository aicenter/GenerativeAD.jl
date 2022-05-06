using DrWatson
@quickactivate
using BSON, FileIO
using NPZ

dataset = "CIFAR10"
ac = 7
seed = 1
modelname = "sgvae"




score = "normal_logpx"
model_id = 86500028
model_id = 87940842
model_id = 11939888
model_id = 41159436


f = joinpath(indir, "model_id=$(model_id)_score=$(score).bson")

data = load(f)

outdir = datadir("sgad_latent_scores/images_leave-one-in_converted/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)/model_id=$(model_id)_score=$(score)")
mkpath(outdir)

for (s, l) in zip(["train", "validation", "test"], [:tr_scores, :val_scores, :tst_scores])
	outf = joinpath(outdir, "$(s).npy")
	npzwrite(outf, data[l])
end

in_f = "/home/skvara/work/counterfactual_ad/GenerativeAD.jl/data/experiments/images_leave-one-in/sgvae/CIFAR10/ac=7/seed=1/batch_size=128_fixed_mask_epochs=3_h_channels=16_img_dim=32_init_gain=0.02_init_seed=86500028_init_type=orthogonal_lr=0.00025_score=logpx_tau_mask=0.1_version=0.2_weight_binary=50_weight_mask=1000_z_dim=64.bson"
in_f = "/home/skvara/work/counterfactual_ad/GenerativeAD.jl/data/experiments/images_leave-one-in/sgvae/CIFAR10/ac=7/seed=1/batch_size=64_fixed_mask_epochs=2_h_channels=64_img_dim=32_init_gain=0.02_init_seed=87940842_init_type=normal_lr=0.0002_score=logpx_tau_mask=0.3_version=0.2_weight_binary=1_weight_mask=0.5_z_dim=128.bson"
in_f = "/home/skvara/work/counterfactual_ad/GenerativeAD.jl/data/experiments/images_leave-one-in/sgvae/CIFAR10/ac=7/seed=1/batch_size=32_fixed_mask_epochs=0_h_channels=8_img_dim=32_init_gain=0.06_init_seed=11939888_init_type=normal_latent_structure=independent_lr=0.00032_score=logpx_tau_mask=0.2_weight_binary=0.1_weight_mask=0_z_dim=32.bson"
in_f = "/home/skvara/work/counterfactual_ad/GenerativeAD.jl/data/experiments/images_leave-one-in/sgvae/CIFAR10/ac=7/seed=1/batch_size=128_fixed_mask_epochs=1_h_channels=16_img_dim=32_init_gain=0.07_init_seed=41159436_init_type=orthogonal_lr=0.0004_score=logpx_tau_mask=0.2_version=0.2_weight_binary=0.05_weight_mask=0.05_z_dim=64.bson"

data = load(in_f)
outdir = "/home/skvara/work/counterfactual_ad/sgad/notebooks/$(model_id)_data/"
mkpath(outdir)
for (s, l) in zip(["train", "validation", "test"], [:tr_scores, :val_scores, :tst_scores])
	outf = joinpath(outdir, "$(s)_logpx_scores.npy")
	npzwrite(outf, data[l])
end
