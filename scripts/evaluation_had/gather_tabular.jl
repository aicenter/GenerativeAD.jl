using DrWatson
using DataFrames, BSON, FileIO

had_bson = datadir(ARGS[1])
original_bson = datadir(ARGS[2])
new_bson = datadir(ARGS[3])

hdf = load(had_bson)[:df]
odf = load(original_bson)[:df]
ndf = vcat(odf, hdf)
save(new_bson, Dict(:df => ndf))