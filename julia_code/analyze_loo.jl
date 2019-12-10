using CSV
using DataFrames
using FileIO
using JLD2
using HDF5
using Printf
using Statistics
using PyPlot
using LaTeXStrings

USE_ACCEPTED_LAMBDAS = true

henry_df = CSV.read(joinpath("..","data","henry_matrix_df_l_4.csv"))
gases = names(henry_df)[2:end]
materials = henry_df[:,1]

H = convert(Array{Union{Float64, Missing}, 2}, henry_df[1:end, 2:end])
log_H = log10.(H)

# Each jld file has three variables, :loss, :err and :paritypred
jld_files = [file for file in readdir("results") if split(file, ".")[end] == "jld2"]
rs = unique([z[2] for z in split.(jld_files, "_")])

accepted_lambda_ls = 10 .^ range(2, 4, length=20)
accepted_lambda_bs = 10 .^ range(0, 2, length=20)

unique_lambda_ls = USE_ACCEPTED_LAMBDAS ? Dict(r => accepted_lambda_ls for r in rs) : Dict(r => sort!(unique([parse(Float64, z[3]) for z in split.(jld_files, "_") if z[2] == r])) for r in rs)
unique_lambda_bs = USE_ACCEPTED_LAMBDAS ? Dict(r => accepted_lambda_bs for r in rs) : Dict(r => sort!(unique([parse(Float64, split(z[4], ".jld2")[1]) for z in split.(jld_files, "_") if z[2] == r])) for r in rs)
lowest_err = Dict(r => Inf for r in rs)
lowest_err_filename = Dict(r => "" for r in rs)
loo_pred = Dict(r => zeros(size(log_H)) for r in rs)
loo_rmses = Dict(r => fill(Inf, (length(unique_lambda_ls[r]), length(unique_lambda_bs[r]))) for r in rs)
println(size(loo_rmses["0"]))

for jld_file in jld_files
    r = split(jld_file, "_")[2]
    lambda_l = parse(Float64, split(jld_file, "_")[3])
    lambda_b = parse(Float64, split(split(jld_file, ".jld2")[1], "_")[4])
    try
        l_idx = findall(isapprox.(lambda_l, unique_lambda_ls[r], rtol=1e-3))[1]
        b_idx = findall(isapprox.(lambda_b, unique_lambda_bs[r], rtol=1e-3))[1]
        #@printf("Working on %s\n", jld_file)
        global lowest_err
        global lowest_err_filename
        global loo_pred
        results = load("results/" * jld_file)
        test_rmse = results["test_rmse"]
        loo_rmses[r][l_idx, b_idx] = test_rmse
        if test_rmse < lowest_err[r]
            lowest_err[r] = test_rmse
            lowest_err_filename[r] = jld_file
            loo_pred[r] = results["H_LOO_prediction"]
        end
    catch
        continue
    end
end

for r in rs
    fig, ax = plt.subplots(figsize=(14,12))
    for x in 1:size(loo_rmses[r])[2]
        ax.plot([x, x], [0, size(loo_rmses[r])[1]], linewidth=1.25, color="k")
    end
    for y in 1:size(loo_rmses[r])[1]
        ax.plot([0, size(loo_rmses[r])[2]], [y, y], linewidth=1.25, color="k")
    end

    img = ax.pcolormesh(loo_rmses[r], cmap="viridis", vmin=0.756, vmax=0.8)
    extend_string = any(loo_rmses[r][:] .> 0.9) ? "max" : "neither"
    cbar = plt.colorbar(img, extend=extend_string)
    cbar.set_label("Test RMSE", fontsize=14, rotation=270, labelpad=25)
    ax.set_xlabel(L"$\lambda_b$", fontsize=14)
    ax.set_ylabel(L"$\lambda_l$", fontsize=14)
    ax.set_xticks(1:length(unique_lambda_bs[r]))
    ax.set_xticklabels([@sprintf("%.2e", x) for x in unique_lambda_bs[r]], rotation=90)
    ax.set_yticklabels([@sprintf("%.2e", x) for x in unique_lambda_ls[r]])
    ax.set_title(@sprintf("r = %d", parse(Int, r)), fontsize=16)
    ax.set_yticks(1:length(unique_lambda_ls[r]))


    plt.tight_layout()
    plt.savefig("loss_heatmap_r" * r * ".png", format="png", dpi=300)
    plt.close()
end

for _r in rs
    _, r, λl, λb = split(split(lowest_err_filename[_r], ".jld2")[1], "_")
    @printf("Best model is r = %d, lambda_l = %.3f, lambda_b = %.3f\n", 
                parse(Int, r), parse(Float64, λl), parse(Float64, λb))

    prediction = loo_pred[_r]
    actual = log_H[:]

    non_missing_indices = .!ismissing.(actual)
    actual_mean = mean(actual[non_missing_indices])
    R² = 1 - (sum([(actual[i] - prediction[i])^2 for i = collect(1:length(actual))[non_missing_indices]]) / sum([(actual[i] - actual_mean)^2 for i = collect(1:length(actual))[non_missing_indices]]))
    @printf("R2 = %.3f\n", R²)
end
