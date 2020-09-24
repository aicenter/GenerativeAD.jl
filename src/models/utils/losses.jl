_init_logJ(data) = zeros(eltype(data), 1, size(data, 2))

sqnorm(x) = sum(abs2, x)
l2_reg(ps) = sum(sqnorm, ps)
