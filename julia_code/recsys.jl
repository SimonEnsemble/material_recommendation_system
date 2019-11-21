using CSV
using DataFrames
using Printf
using JSON
using Statistics
using LinearAlgebra
using JLD2
using FileIO
include("recsys_funcs.jl")


henry_df = CSV.read(joinpath("..","data","henry_matrix_df_l_4.csv"))
gases = names(henry_df)[2:end]
materials = henry_df[:,1]

H = convert(Array{Union{Float64, Missing}, 2}, henry_df[1:end, 2:end])

_r = [1, 2]
_λ₁ = 10 .^range(1, 3, length=10)
_λ₂ = 10 .^range(-1, 1, length=10)

for r in _r
    for λ₁ in _λ₁
        for λ₂ in _λ₂
			LOO_for_cluster(H, r, λ₁, λ₂)
        end
    end
end
