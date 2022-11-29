include("../supervised_comparison/utils.jl")
include("./utils.jl")

s = ArgParseSettings()
@add_arg_table! s begin
	"modelname"
        default = "sgvaegan100"
		arg_type = String
		help = "model name"
    "dataset"
        default = "wildlife_MNIST"
        arg_type = String
        help = "dataset"
    "anomaly_class"
    	default = 1
        arg_type = Int
        help = "anomaly class"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, anomaly_class, force = parsed_args
ac = anomaly_class

model_id = 32131929

ldir = datadir("experiments_multifactor/latent_scores/$modelname/$dataset/ac=$ac/seed=1")
lfs = readdir(ldir)
model_ids = map(x->Meta.parse(replace(split(x,"_")[2], "id="=>"")), lfs)

modelinds = model_ids .== model_id
final_model_ids = model_ids[modelinds]
final_lfs = lfs[modelinds]

i = 1
lf = final_lfs[i]
model_id = final_model_ids[i]

ldata = load(joinpath(ldir, lf))
scores = ldata[:mf_scores]
mf_labels = ldata[:mf_labels]

# first a small experiment - anomalies are in the shape
anomalous_inds = 
true_inds = map(i->mf_labels[i,:] .== ac, 1:3)
y_ano1 = 


using Random
mf_labels = hcat([sample(1:10,3) for _ in 1:60000]...)
mf_scores = rand(3,60000)
tr_scores = rand(3,1000)
ac = 1

s = mf_scores[:,1]
af = 1

afs = [1,2,3]
@assert(af in afs)
nafs = afs[afs.!=af]

# anomalies in shape
inds = .!true_inds[af] .& true_inds[nafs[1]] .& true_inds[nafs[2]]
sublabels = mf_labels[:,inds]
subscores = mf_scores[:,inds]
s = subscores[:,1]

predict_anomaly_factor(tr_scores, subscores[:,3])

n = size(subscores, 2)
y_true = ones(Int, n)*af
y_pred = mapslices(x->predict_anomaly_factor(tr_scores,x), subscores, dims=1)
acc = mean(y_true .== y_pred)