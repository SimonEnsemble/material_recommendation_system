using PyPlot
using CSV
using DataFrames
using Printf
using Statistics
using LinearAlgebra
include("recsys_funcs.jl")


henry_df = CSV.read(joinpath("..","data","henry_matrix_df.csv"))
gases = names(henry_df)[2:end]
materials = henry_df[:,1]

H = convert(Array{Union{Float64, Missing}, 2}, henry_df[2:end, 2:end])
H[83,8] = missing
log_H = log10.(H)
#M, G, train_error = ALS(log_H, 5, [2.0, 0.01], 1e-5, maxiter=50000)

fold_matrix = k_fold_split(log_H, 3)

_r = [4,5]
_λ₁ = [0.1, 1.0, 2.0]
_λ₂ = [0.01, 0.1, 1.0]
n_cv = 0
CV_dict = Dict{AbstractString, Any}()

for r in _r
    for λ₁ in _λ₁
        for λ₂ in _λ₂
            global n_cv += 1
#            test_error, parity_pred = cross_validation(log_H, fold_matrix, r, λ₁, λ₂)
            test_error, parity_pred = LOO_cross_validation(log_H, r, λ₁, λ₂)
            CV_dict[string(n_cv) * "_err"] = test_error
            CV_dict[string(n_cv) * "_pred"] = parity_pred
            CV_dict[string(n_cv) * "_lambda1"] = λ₁
            CV_dict[string(n_cv) * "_lambda2"] = λ₂
            CV_dict[string(n_cv) * "_r"] = r
        end
    end
end

