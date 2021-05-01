###
# using Revise
###
using DrWatson
using GenerativeAD
using StatsBase: fit!, predict, sample
using Statistics
using BSON
using Flux
using NPZ
using Random

parameters = (
    lambda_rat 	= 1,
    lr 			= 1f-4,
    batchsize 	= 1024,
    wreg 		= 0.0,
)

# loading KDDCUP data from npz file
file = datadir("kdd_cup.npz") # taken from https://github.com/mperezcarrasco/PyTorch-DAGMM
pydata = NPZ.npzread(file)

labels = pydata["kdd"][:, end];
features = copy(pydata["kdd"][:, 1:end-1]')

normal_data = features[:, labels .== 0];
anomalous_data = features[:, labels .== 1];

# split data with fixed seed
data = GenerativeAD.Datasets.train_val_test_split(normal_data, anomalous_data; seed=1)
idim = size(data[1][1], 1)

# when running in batch randomize model seed (reset after training - deterministic random batches)
model_seed = rand(1:1000)
Random.seed!(model_seed);
encoder = Chain(Dense(idim, 60, tanh), Dense(60, 30, tanh), Dense(30, 10, tanh), Dense(10, 1))
decoder = Chain(Dense(1, 10, tanh), Dense(10, 30, tanh), Dense(30, 60, tanh), Dense(60, idim))
estimator = Chain(Dense(3, 10, tanh), Dropout(0.5), Dense(10, 4))

model = GenerativeAD.Models.DAGMM(encoder, decoder, estimator)
info, fit_t, _, _, _ = @timed fit!(model, data; patience=20, check_interval=10, parameters...)
Random.seed!()

trained_model = info.model

testmode!(trained_model, true)
_, _, z, gamma = trained_model(data[1][1])
phi, mu, cov = GenerativeAD.Models.compute_params(z, gamma)
testmode!(trained_model, false)

trn_scores = predict(trained_model, data[1][1], phi, mu, cov);
val_scores = predict(trained_model, data[2][1], phi, mu, cov);
tst_scores = predict(trained_model, data[3][1], phi, mu, cov);

using EvalMetrics: ConfusionMatrix, recall, precision, f1_score, roccurve, auc_trapezoidal

# test only
threshold = quantile(vcat(trn_scores, val_scores, tst_scores), 0.8)
# threshold = quantile(vcat(trn_scores, val_scores), 0.8)

tst_ŷ = tst_scores .> threshold;
cm = ConfusionMatrix(data[3][2], tst_ŷ)
rec, prc, f1s = recall(cm), precision(cm), f1_score(cm)

roc = roccurve(data[3][2], tst_scores)
auc = auc_trapezoidal(roc)


# test + validation
# val_ŷ = val_scores .> threshold;
# cm = ConfusionMatrix(vcat(data[2][2], data[3][2]), vcat(val_ŷ, tst_ŷ))
# recall(cm), precision(cm), f1_score(cm)

results = Dict(
    :history    => info.history,
    :niter      => info.niter,
    :model      => info.model,
    :recall     => rec,
    :precision  => prc,
    :f1         => f1s,
    :auc        => auc,
    :phi        => phi, 
    :mu         => mu,
    :cov        => cov,
    :threshold  => threshold,
    :trn_scores => trn_scores,
    :trn_labels => data[1][2],
    :val_scores => val_scores,
    :val_labels => data[2][2],
    :tst_scores => tst_scores,
    :tst_labels => data[3][2]
)

@info("", model_seed, threshold, rec, prc, f1s, auc)

save(datadir("dumpster/dagmm_kddcup99_$(model_seed).bson"), results)


### analysis
using DataFrames, ValueHistories
files = readdir(datadir("dumpster"), join=true)
results = [load(f) for f in files]

df = DataFrame(
    "model_seed" => [parse(Int, split(split(f, '_')[end], '.')[1]) for f in files],
    "niter" => [r[:niter] for r in results],
    "precision" => [r[:precision] for r in results],
    "recall" => [r[:recall] for r in results],
    "f1" => [r[:f1] for r in results],
    "auc" => [r[:auc] for r in results],
    "threshold" => [r[:threshold] for r in results],
)

sort!(df, :niter)
sort!(df, :auc)

###