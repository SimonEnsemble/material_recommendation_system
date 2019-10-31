using CSV
using DataFrames
using JSON
using Printf
using PyPlot

cv_dicts = readdir("cv_results")
rs = Array{Int, 1}()
errs = Array{Float64, 1}()
lambda1s = Array{Float64, 1}()
lambda2s = Array{Float64, 1}()

for cv_dict in cv_dicts
    @printf("Reading %s\n", cv_dict)
    infile = open("cv_results/" * cv_dict, "r")
    lines = readlines(infile)
    close(infile)
    json_data = JSON.parse(lines[1])
    min_err = Inf
    min_key = 0
    for i = 1:Int(length(json_data)/5)
        if json_data[string(i) * "_err"] < min_err
            min_err = json_data[string(i) * "_err"]
            min_key = i
        end
        append!(rs, json_data[string(i) * "_r"])
        append!(errs, json_data[string(i) * "_err"])
        append!(lambda1s, json_data[string(i) * "_lambda1"])
        append!(lambda2s, json_data[string(i) * "_lambda2"])
    end
    min_l1 = json_data[string(min_key) * "_lambda1"]
    min_l2 = json_data[string(min_key) * "_lambda2"]
    min_r = json_data[string(min_key) * "_r"]
    @printf("r: %d\nlambda1: %.3f\nlambda2: %.3f\nErr: %.3f\n---------------------------------\n", min_r, min_l1, min_l2, min_err)
end
