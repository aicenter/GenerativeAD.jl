using DrWatson
@quickactivate
using PrettyTables
using Statistics
using FileIO, BSON, DataFrames

#DrWatson.datadir(path) = joinpath("/home/skvarvit/generativead/GenerativeAD.jl/data", path)
df_tabular = load(datadir("evaluation/tabular_eval.bson"))[:df];

scores = ["reconstruction-sampled", "reconstruction-mean", "jacodeco"]
models = ["vae", "vae_full", "vae_simple", "aae", "aae_full", "wae", "wae_full"]

mean_scores = Dict()
for model in models
	mdf = filter(r->r.modelname == model, df_tabular)
	mean_scores[model] = []
	for score in scores
		subdf = filter(r-> occursin(score, r.parameters), mdf)
		push!(mean_scores[model], mean(subdf.tst_eval_t))
	end
end
mean_scores
df_out = DataFrame()
df_out[!,:model] = models
for (i,score) in enumerate(scores)
	df_out[!,Symbol(score)] = map(m->mean_scores[m][i],models)
end

println(df_out)