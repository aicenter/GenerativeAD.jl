using Revise

using DrWatson
using GenerativeAD
using StatsBase: fit!, predict, sample
using Statistics
using BSON
using Flux
using DataFrames, CSV
using Random
using EvalMetrics: ConfusionMatrix, recall, precision, f1_score, roccurve, auc_trapezoidal

for dataset in ["probe", "secom", "u2r"]
    for subsample in 2 .^(3:8)
        parameters = (
            confidence_margin  = 1000.0,
            batchsize 	       = 256,
            zdim               = 20,
            nlayers            = 1,
            hdim               = 10, # does not matter with one layer
            activation         = "relu",
            ensemble_size      = 50,
            subsample_size     = 16,
            init_seed          = 42
        )

        parameters = merge(parameters, (;subsample_size=subsample))

        # loading data from RDP paper https://github.com/billhhh/RDP/tree/master/RDP-Anomaly/data
        file = datadir("rdp_datasets/$(dataset).csv")
        df = CSV.read(file);

        labels = Int.(df[:class]);
        features = Float32.(copy(Matrix(select(df, Not(:class)))'));

        data = ((features, labels),)
        idim = size(data[1][1], 1)

        @info "Started training REPEN$(parameters) on $(dataset)"
        @info "Number of data points : $(size(data[1][1], 2))"
        @info "Number of features: $(size(data[1][1], 1))"

        model = GenerativeAD.Models.REPEN(;idim=idim, parameters...)
        try
            global info, fit_t, _, _, _ = @timed fit!(model, data; parameters...)
        catch e
            @warn "Training failed due to $e. Continuing."
            continue
        end

        lesinn_trn_scores = GenerativeAD.Models.lesinn(data[1][1]; fit_data=data[1][1], parameters...);
        trn_scores = predict(model, data[1][1]; fit_data=data[1][1], parameters...);

        auc = roccurve(data[1][2], trn_scores) |> auc_trapezoidal
        lesinn_auc = roccurve(data[1][2], lesinn_trn_scores) |> auc_trapezoidal

        @info("", dataset, subsample, auc, lesinn_auc)

        results = Dict(
            :dataset        => dataset,
            :modelname      => "REPEN",
            :history        => info.history,
            :model          => info.model,
            :auc            => auc,
            :lesinn_auc     => lesinn_auc,
            :trn_scores     => trn_scores,
            :trn_labels     => data[1][2],
            :parameters     => parameters,
            :subsample_size => parameters.subsample_size
        )

        wsave(datadir("dumpster/repen/$(dataset)_$(subsample).bson"), results)
    end
end

### analysis
using DataFrames, ValueHistories, CSV.SentinelArrays
files = readdir(datadir("dumpster/repen"), join=true)
results = [load(f) for f in files]

df = DataFrame(
    "dataset" => [r[:dataset] for r in results],
    "subsample" => [r[:subsample_size] for r in results],
    "auc" => [r[:auc] for r in results],
    "lesinn_auc" => [r[:lesinn_auc] for r in results]
)

sort(df, (:dataset, :auc))
