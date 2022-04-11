#!/bin/bash
julia --project compute_knn.score.jl sgvae wildlife_MNIST leave-one-in
julia --project compute_knn.score.jl sgvae CIFAR10 leave-one-in
julia --project compute_knn.score.jl sgvae SVHN2 leave-one-in

julia --project compute_knn.score.jl sgvae wood mvtec
julia --project compute_knn.score.jl sgvae grid mvtec
julia --project compute_knn.score.jl sgvae bottle mvtec
julia --project compute_knn.score.jl sgvae pill mvtec
julia --project compute_knn.score.jl sgvae capsule mvtec
julia --project compute_knn.score.jl sgvae metal_nut mvtec
julia --project compute_knn.score.jl sgvae transistor mvtec