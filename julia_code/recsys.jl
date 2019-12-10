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

# LOO cross-valiation procedure
_r = [3]
_λ₁ = 10 .^range(2, 4, length=20)
_λ₂ = 10 .^range(0, 2, length=20)

for r in _r
    for λ₁ in _λ₁
        for λ₂ in _λ₂
            # submit LOO job
            if submit_LOO_jobs
				println("this is true")
				submit_filename = write_submit_file(H, r, λ₁, λ₂)
				# If we've already ran these calcs, skip
				if any(occursin.(split(submit_filename, ".sh")[1] * ".jld2", readdir("results")))
					@printf("We have already calculated the LOO cross validation for this file. Skipping\n")
					continue
			    end
				run(`qsub $submit_filename`)
				sleep(0.25)
				run(`rm -f $submit_filename`)
            # collect results from LOO job
            else
				println("this is false")
				continue
                #@load filename_results
            end
        end
    end
end

