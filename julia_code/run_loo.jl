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

henry_df = CSV.read(joinpath("..", "data", "henry_matrix_df_l_4.csv"))
H = convert(Array{Union{Float64, Missing}, 2}, henry_df[1:end, 2:end])
log_H = log10.(H)

test_rmse, H_LOO_prediction = LOO_cross_validation(log_H, r, λ₁, λ₂, "results/" * loo_filename(r,λ₁,λ₂) * ".jld2";
												   min_als_sweeps=30, max_als_sweeps=100)
