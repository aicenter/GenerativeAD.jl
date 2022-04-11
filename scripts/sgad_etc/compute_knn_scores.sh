#!/bin/bash
julia --project compute_knn_scores.jl sgvae wildlife_MNIST leave-one-in
julia --project compute_knn_scores.jl sgvae CIFAR10 leave-one-in
julia --project compute_knn_scores.jl sgvae SVHN2 leave-one-in

julia --project compute_knn_scores.jl sgvae wood mvtec
julia --project compute_knn_scores.jl sgvae grid mvtec
julia --project compute_knn_scores.jl sgvae bottle mvtec
julia --project compute_knn_scores.jl sgvae pill mvtec
julia --project compute_knn_scores.jl sgvae capsule mvtec
julia --project compute_knn_scores.jl sgvae metal_nut mvtec
julia --project compute_knn_scores.jl sgvae transistor mvtec