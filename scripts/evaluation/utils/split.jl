# split ocsvm to rbf/nu variants
function split_ocsvm(df)
    orbf_df = filter(x -> (x.modelname == "ocsvm") & occursin("rbf", x.parameters), df)
    orbf_df[:modelname] .= "ocsvm_rbf"

    osnu_df = filter(x -> (x.modelname == "ocsvm") & occursin("nu=0.01", x.parameters), df)
    osnu_df[:modelname] .= "ocsvm_n1"

    vcat(df, orbf_df, osnu_df)
end

# function split_autoencoders(df)
# end