function setup_classic_models(df)
    # rename datasets
    apply_aliases!(df, col="dataset", d=DATASET_ALIAS) # rename
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

function differentiate_beta_1_10(df)
    # sgvaeganalpha - beta=1/10
    subdfa = filter(r->r.modelname == "sgvaegan_alpha", df)
    parametersa = map(x->parse_savename(x)[2], subdfa.parameters)
    subdfa.modelname[[x["beta"] for x in parametersa] .== 1.0] .= "sgvaegan_alpha_1"
    subdfa.modelname[[x["beta"] for x in parametersa] .== 10.0] .= "sgvaegan_alpha_10"
    df = vcat(df, subdfa)
    MODEL_ALIAS["sgvaegan_alpha_1"] = "sgvgna_b1"
    MODEL_ALIAS["sgvaegan_alpha_10"] = "sgvgna_b10"
    df
end

function differentiate_early_stopping(df)
    # sgvaegan/vaegan/fmganpy - 1000 or 10 early stopping anomalies
    subdf = filter(r->r.modelname in ["sgvaegan", "vaegan", "fmganpy"], df)
    parameters = map(x->parse_savename(x)[2], subdf.parameters)
    vs = [get(x, "version", 0.3) for x in parameters]
    subdf.modelname[vs .== 0.3] .= subdf.modelname[vs .== 0.3] .* "_0.3"
    subdf.modelname[vs .== 0.4] = subdf.modelname[vs .== 0.4] .* "_0.4"
    df = vcat(df, subdf)
    MODEL_ALIAS["sgvaegan_0.3"] = "sgvgn03"
    MODEL_ALIAS["vaegan_0.3"] = "vgn03"
    MODEL_ALIAS["fmganpy_0.3"] = "fmgn03"
    MODEL_ALIAS["sgvaegan_0.4"] = "sgvgn04"
    MODEL_ALIAS["vaegan_0.4"] = "vgn04"
    MODEL_ALIAS["fmganpy_0.4"] = "fmgn04"
    df
end

function differentiate_sgvaegana(df)
    # also add sgvaegan alpha - 0.3/0.4
    subdfa = filter(r->r.modelname == "sgvaegan_alpha", df)
    parametersa = map(x->parse_savename(x)[2], subdfa.parameters)
    subdfa.modelname[[get(x, "version", 0.3) for x in parametersa] .== 0.3] .= "sgvaegan_alpha_0.3"
    subdfa.modelname[[get(x, "version", 0.3) for x in parametersa] .== 0.4] .= "sgvaegan_alpha_0.4"
    df = vcat(df, subdfa)
    MODEL_ALIAS["sgvaegan_alpha_0.3"] = "sgvgna03"
    MODEL_ALIAS["sgvaegan_alpha_0.4"] = "sgvgna04"
    df
end