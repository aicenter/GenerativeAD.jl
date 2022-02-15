using FileIO, BSON, DataFrames

f_new = "/home/skvarvit/generativead-sgad/GenerativeAD.jl/data/evaluation/images_leave-one-in_eval.bson"
f_old = "/home/skvarvit/generativead/GenerativeAD.jl/data/evaluation/images_leave-one-in_eval.bson"
f_target = "/home/skvarvit/generativead-sgad/GenerativeAD.jl/data/evaluation/images_leave-one-in_eval_all.bson"

d_new = load(f_new)
d_old = load(f_old)

df = vcat(d_old[:df], d_new[:df])

save(f_target, Dict(:df => df))

# now repeat for mvtec data
f_new = "/home/skvarvit/generativead-sgad/GenerativeAD.jl/data/evaluation/images_mvtec_eval.bson"
f_old = "/home/skvarvit/generativead/GenerativeAD.jl/data/evaluation/images_mvtec_eval.bson"
f_target = "/home/skvarvit/generativead-sgad/GenerativeAD.jl/data/evaluation/images_mvtec_eval_all.bson"

d_new = load(f_new)
d_old = load(f_old)

df = vcat(d_old[:df], d_new[:df])

save(f_target, Dict(:df => df))
