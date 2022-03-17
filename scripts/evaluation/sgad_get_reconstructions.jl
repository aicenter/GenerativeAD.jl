# use this to collect the images of reconstructed data for the lastest iterations of the sgvae model
using DrWatson
@quickactivate
using GenerativeAD

function find_copy_latest_reconstruction(dir, outpath)
    # construct outdir
    modelid = split(dir, "/")[end-1]
    seed = split(dir, "/")[end-2]
    ac = split(dir, "/")[end-3]

    # select the right file
    fs = filter(x->occursin("reconstructed_mean", x), readdir(dir))
    maxi = argmax(map(x->Meta.parse(split(x, "_")[2]), fs))
    f = joinpath(dir, fs[maxi])
    outf = joinpath(outpath, "$(ac)_$(seed)_$(modelid)_$(fs[maxi])")
    cp(f, outf, force=true)
end

function copy_latest_reconstructions(model, datatype, dataset)
    master_path = datadir("sgad_models/images_$(datatype)/$(model)/$(dataset)")
    outpath = datadir("sgad_outputs/images_$(datatype)/$(model)/$(dataset)")
    mkpath(outpath)
    samples = GenerativeAD.Evaluation.collect_files(master_path)
    filter!(x->occursin("reconstructed_mean", x), samples)
    superfs = unique(dirname.(samples))

    for dir in superfs
        find_copy_latest_reconstruction(dir, outpath)
    end
end

copy_latest_reconstructions("sgvae", "leave-one-in", "wildlife_MNIST")
copy_latest_reconstructions("sgvae", "leave-one-in", "CIFAR10")
copy_latest_reconstructions("sgvae", "leave-one-in", "SVHN2")
copy_latest_reconstructions("sgvae", "mvtec", "SVHN2")
