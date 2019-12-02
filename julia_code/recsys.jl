using CSV
using DataFrames
using Printf
using JSON
using Statistics
using LinearAlgebra
using JLD2
using FileIO
include("recsys_funcs.jl")

# directions
submit_LOO_jobs = true

## build data
henry_df = CSV.read(joinpath("..","data","henry_matrix_df_l_4.csv"))
gases = names(henry_df)[2:end]
materials = henry_df[:,1]

H = convert(Array{Union{Float64, Missing}, 2}, henry_df[1:end, 2:end])
log_H = log10.(H)

## LOO cross-valiation procedure
_r = [0, 1, 2]
_λ₁ = 10 .^range(1, 3, length=10)
_λ₂ = 10 .^range(-1, 1, length=10)

for r in _r
    for λ₁ in _λ₁
        for λ₂ in _λ₂
            # where are LOO results saved to JLD2?
            filename_results = LOOfilename(r, λ₁, λ₂)
            # submit LOO job
            if submit_LOO_jobs
                test_rmse, log_H_pred = LOO_cross_validation(log_H, r, λ₁, λ₂, filename_results,
                                                             min_als_sweeps=25, max_als_sweeps=500)
            # collect results from LOO job
            else
                @load filename_results
            end
        end
    end
end

