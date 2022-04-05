using DrWatson
@quickactivate
using GenerativeAD
using BSON, FileIO, DataFrames

function row(d)
    ps = d[:parameters]
    return  [
        d[:modelname],
        ps[:init_seed],
        d[:dataset],
        d[:anomaly_class],
        d[:seed],
        ps[:weights_texture],
        ps[:detach_mask]
        ]
end

model = "sgvae"
datatype = "leave-one-in"
mainpath = datadir("experiments/images_$(datatype)/$(model)")

outdf = DataFrame(
    :modelname => [],
    :init_seed => [],
    :dataset => [],
    :anomaly_class => [],
    :seed => [],
    :weights_texture => [],
    :detach_mask => []
    )


for dataset in ["wildlife_MNIST", "CIFAR10", "SVHN2"]
    for ac in 1:10
        for seed in 1:1
            inpath = joinpath(mainpath, "$(dataset)/ac=$(ac)/seed=$(seed)")
            fs = filter(x-> !occursin("model", x),readdir(inpath, join=true))
            for f in fs
                d = load(f)
                push!(outdf, row(d))
            end
            @info "Finished processing $inpath"
        end
    end
end

outdir = datadir("sgad_outputs/vector_params")
mkdir(outdir)
outf = joinpath(outdir, "images_$(datatype).bson")
save(outf, Dict(:df => outdf))

# mvtec
datatype = "leave-one-in"
mainpath = datadir("experiments/images_$(datatype)/$(model)")

outdf = DataFrame(
    :modelname => [],
    :init_seed => [],
    :dataset => [],
    :anomaly_class => [],
    :seed => [],
    :weights_texture => [],
    :detach_mask => []
    )


for dataset in ["wood", "grid", "transistor", "pill", "metal_nut", "capsule", "bottle"]
    for ac in 1:1
        for seed in 1:5
            inpath = joinpath(mainpath, "$(dataset)/ac=$(ac)/seed=$(seed)")
            fs = filter(x-> !occursin("model", x),readdir(inpath, join=true))
            for f in fs
                d = load(f)
                push!(outdf, row(d))
            end
            @info "Finished processing $inpath"
        end
    end
end

outdir = datadir("sgad_outputs/vector_params")
mkdir(outdir)
outf = joinpath(outdir, "images_$(datatype).bson")
save(outf, Dict(:df => outdf))
