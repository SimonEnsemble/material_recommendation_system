using CSV
using DataFrames
using Printf
using JSON
using Statistics
using LinearAlgebra
using JLD2
using FileIO
include("recsys_funcs.jl")

r = parse(Int, ARGS[1])
λ₁ = parse(Float64, ARGS[2])
λ₂ = parse(Float64, ARGS[3])
idx1 = parse(Int, ARGS[4])
idx2 = parse(Int, ARGS[5])

henry_df = CSV.read(joinpath("..", "data", "henry_matrix_df_l_4.csv"))
H = convert(Array{Union{Float64, Missing}, 2}, henry_df[1:end, 2:end])
log_H = log10.(H)
log_H[idx1, idx2] = missing

M, G, mu, gamma, err, loss, hbar = ALS(log_H, r, [λ₁, λ₂], 1e-6, 1e-7, 10000, false)

H_pred = M' * G .+ hbar .+ mu' .+ gamma
parity_pred = H_pred[idx1, idx2]

result_file = open(@sprintf("results/LOO_%d_%.3f_%.3f_%d_%d.csv", r, λ₁, λ₂, idx1, idx2), "w")
@printf(result_file, "err,loss,parpred\n%.6f,%.6f,%.6f", err, loss, parity_pred)
close(result_file)

