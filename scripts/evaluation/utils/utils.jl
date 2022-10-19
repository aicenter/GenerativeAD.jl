function setup_classic_models(df)
    # rename datasets
    apply_aliases!(df_images, col="dataset", d=DATASET_ALIAS) # rename
    # no sgvae alpha here
    df = filter(r->r.modelname != "sgvae_alpha", df);
    # only use (sg)vaegan with disc score
    df = filter(r->!(r.modelname in ["sgvaegan10", "sgvaegan100"] && 
        get(parse_savename(r.parameters)[2], "score", "") != "discriminator"), df)
    df = filter(r->!(r.modelname == "vaegan10" && 
        get(parse_savename(r.parameters)[2], "score", "") != "discriminator"), df)
    # also differentiate between the old and new cgn
    df.modelname[map(x->get(parse_savename(x)[2], "version", 0.1) .== 0.2, 
        df.parameters) .& (df.modelname .== "cgn")] .= "cgn_0.2"
    df.modelname[map(x->get(parse_savename(x)[2], "version", 0.1) .== 0.3, 
        df.parameters) .& (df.modelname .== "cgn")] .= "cgn_0.3"
    # finally, set apart the sgvaegan with lin adv score
    df.modelname[map(x->get(parse_savename(x)[2], "version", 0.1) .== 0.5, 
        df.parameters) .& (df.modelname .== "sgvaegan")] .= "sgvaegan_0.5"
    df
end

function prepare_alpha_df!(df)
    df.dataset[df.dataset .== "metal_nut"] .= "nut"
    df["fs_fit_t"] = NaN
    df["fs_eval_t"] = NaN
    df
end

function setup_alpha_models(df)
    df = filter(r->occursin("_robreg", r.modelname), df)
    df = filter(r->get(parse_savename(r.parameters)[2], "beta", 1.0) in [1.0, 10.0], df)
    for model in ["sgvae_", "sgvaegan_", "sgvaegan10_", "sgvaegan100_"]
        df.modelname[map(r->occursin(model, r.modelname), eachrow(df))] .= model*"alpha"
    end
    apply_aliases!(df, col="dataset", d=DATASET_ALIAS) # rename
    prepare_alpha_df!(df)
    df
end