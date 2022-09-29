using DrWatson
@quickactivate
using DataFrames, BSON, FileIO
using StatsBase, CSV
using GenerativeAD
using PyPlot
using GenerativeAD.Evaluation: _prefix_symbol
using EvalMetrics
using Random
using ImageTransformations
using ImageMagick
using Images

include("../scripts/pyutils.jl")
# so the we dont get the "too many open files" os error
torch = pyimport("torch")
torch.multiprocessing.set_sharing_strategy("file_system")
device = "cpu"
modelname = "cgn"
dataset = "SVHN2"
dtst = "svhn"

outdir = datadir("imgs/mask_eval")
mkpath(outdir)

auc_val(labels, scores) = EvalMetrics.auc_trapezoidal(EvalMetrics.roccurve(labels, scores)...)
to_img(x, r=1) = imresize(hcat(map(i->GenerativeAD.Datasets.array_to_img_rgb(permutedims(x[:,:,:,i], (2,1,3)))
            , 1:size(x,4))...), ratio=r)

# setup
acs = 1:10
mn = "AUC"
metric = :auc
val_metric = _prefix_symbol("val", metric)
tst_metric = _prefix_symbol("tst", metric)
AUC_METRICS = ["auc_100", "auc_50", "auc_20", "auc_10", "auc_5", "auc_2", "auc_1", "auc_05", "auc_02", 
    "auc_01"]
AUC_METRICS_NAMES = ["\$AUC@\\%100\$", "\$AUC@\\%50\$", "\$AUC@\\%20\$", "\$AUC@\\%10\$", "\$AUC@\\%5\$", 
    "\$AUC@\\%2\$", "\$AUC@\\%1\$", "\$AUC@\\%0.5\$", "\$AUC@\\%0.2\$", "\$AUC@\\%0.1\$"]
cnames = reverse(AUC_METRICS_NAMES)
level = 100
criterions = reverse(_prefix_symbol.("val", map(x->x*"_$level",  AUC_METRICS)))
extended_criterions = vcat(criterions, [val_metric])
extended_cnames = vcat(["clean"], vcat(cnames, ["\$$(mn)_{val}\$"]))
tst_metrics = vcat(map(x->Symbol(replace(String(x), "val" => "tst")), extended_criterions[1:end-1]), 
    [tst_metric])

# get the data for the best models
df_images = load(datadir("evaluation_kp/images_leave-one-in_eval.bson"))[:df];
subdf = filter(r->r.dataset==dataset && r.modelname==modelname && 
    get(parse_savename(r.parameters)[2], "version", 0.1) == 0.2, 
    df_images)

function gather_results(df, i = 1)
    res = []
    for crit in extended_criterions
        _res = []
        for ac in acs
            _subdf = filter(r->r.anomaly_class == ac && !isnan(r[crit]), df)
            sortinds = sortperm(_subdf[!,crit], rev=true)
            maxind = sortinds[i]
            push!(_res, _subdf[maxind,:])
        end
        push!(res, vcat(map(DataFrame,_res)...))
    end
    res
end
aggregate(dfs) = [mean(d[!,tst_metric]) for d in dfs]
aggregate_alpha(dfs) = map(x->mean(x[1][!,x[2]]), zip(dfs, tst_metrics))
tensor_to_array(x) = permutedims(x.detach().numpy(), (4,3,2,1))

res = gather_results(subdf, 1)

function load_data_and_model(ac)
	# load the original data
	data = GenerativeAD.load_data(dataset, seed=1, anomaly_class_ind=ac, method="leave-one-in");
	data = if modelname != "cgn"
	    GenerativeAD.Datasets.normalize_data(data)
	else
	    data
	end

	mpath = datadir("sgad_models/images_leave-one-in/$(modelname)/$(dataset)/ac=$(ac)/seed=1")
	mfs = readdir(mpath);
	# now load the model
	modelid = parse_savename(res[end][res[end].anomaly_class .== ac,:].parameters[1])[2]["init_seed"]
	mf = joinpath(mpath, filter(x->occursin("$modelid", x), mfs)[1])

	model = GenerativeAD.Models.CGNAnomaly(load_cgn_model(mf, device));
	val_x = data[2][1]
	val_scores = GenerativeAD.Models.predict(model, val_x);
	val_y = data[2][2];
	tst_x = data[3][1]
	tst_scores = GenerativeAD.Models.predict(model, tst_x);
	tst_y = data[3][2];

	val_auc = auc_val(val_y ,val_scores)
	tst_auc = auc_val(tst_y, tst_scores)

	return model, val_x, val_y, val_auc, modelid
end

function generate_save_img(model, val_x, val_y, val_auc, ac, modelid)
	# first select some data to be decomposed
	Random.seed!(1)
	n = 10
	n1 = Int(sum(val_y))
	n0 = length(val_y) - n1
	x_n = val_x[:,:,:,val_y.==0][:,:,:,sample(1:n0, n, replace=false)]
	x_a = val_x[:,:,:,val_y.==1][:,:,:,sample(1:n1, n, replace=false)]
	Random.seed!();

	mask, foreground, background = model.model(torch.tensor(zeros(n)).long())
	mask, foreground, background = map(tensor_to_array, (mask, foreground, background));
	xor = model.model.generate(torch.tensor(zeros(n)).long());
	xor = tensor_to_array(xor);

	# create and save the final image
	imn = to_img(x_n, 2.5)
	imm = to_img(mask, 2.5)
	imf = to_img(foreground, 2.5)
	imb = to_img(background, 2.5)
	xr = mask .* foreground .+ (1 .- mask) .* background
	imr = to_img(xr, 2.5)
	ima = to_img(x_a, 2.5)
	im = vcat(imn, imm, imf, imb, imr, ima)
	f = joinpath(outdir, "$(dataset)_$(ac)_$(modelname)_$(modelid)_$(round(val_auc, digits=3)).png")
	save(f, map(clamp01nan, im))
	@info "saved $f"
end

for ac in 1:10
	model, val_x, val_y, val_auc, modelid = load_data_and_model(ac)
	generate_save_img(model, val_x, val_y, val_auc, ac, modelid)
end
