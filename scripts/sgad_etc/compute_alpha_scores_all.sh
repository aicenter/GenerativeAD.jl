julia --project compute_alpha_scores.jl sgvae wildlife_MNIST leave-one-in normal 
julia --project compute_alpha_scores.jl sgvae wildlife_MNIST leave-one-in kld 
julia --project compute_alpha_scores.jl sgvae wildlife_MNIST leave-one-in normal_logpx 

julia --project compute_alpha_scores.jl sgvae CIFAR10 leave-one-in normal 
julia --project compute_alpha_scores.jl sgvae CIFAR10 leave-one-in kld 
julia --project compute_alpha_scores.jl sgvae CIFAR10 leave-one-in normal_logpx 

julia --project compute_alpha_scores.jl sgvae SVHN2 leave-one-in normal 
julia --project compute_alpha_scores.jl sgvae SVHN2 leave-one-in kld 
julia --project compute_alpha_scores.jl sgvae SVHN2 leave-one-in normal_logpx 

julia --project compute_alpha_scores.jl sgvae bottle mvtec normal 
julia --project compute_alpha_scores.jl sgvae bottle mvtec kld 
julia --project compute_alpha_scores.jl sgvae bottle mvtec normal_logpx 

julia --project compute_alpha_scores.jl sgvae grid mvtec normal 
julia --project compute_alpha_scores.jl sgvae grid mvtec kld 
julia --project compute_alpha_scores.jl sgvae grid mvtec normal_logpx 

julia --project compute_alpha_scores.jl sgvae metal_nut mvtec normal 
julia --project compute_alpha_scores.jl sgvae metal_nut mvtec kld 
julia --project compute_alpha_scores.jl sgvae metal_nut mvtec normal_logpx 

julia --project compute_alpha_scores.jl sgvae transistor mvtec normal 
julia --project compute_alpha_scores.jl sgvae transistor mvtec kld 
julia --project compute_alpha_scores.jl sgvae transistor mvtec normal_logpx 

julia --project compute_alpha_scores.jl sgvae wood mvtec normal 
julia --project compute_alpha_scores.jl sgvae wood mvtec kld 
julia --project compute_alpha_scores.jl sgvae wood mvtec normal_logpx 

julia --project compute_alpha_scores.jl sgvae pill mvtec normal 
julia --project compute_alpha_scores.jl sgvae pill mvtec kld 
julia --project compute_alpha_scores.jl sgvae pill mvtec normal_logpx 

julia --project compute_alpha_scores.jl sgvae capsule mvtec normal 
julia --project compute_alpha_scores.jl sgvae capsule mvtec kld 
julia --project compute_alpha_scores.jl sgvae capsule mvtec normal_logpx 
