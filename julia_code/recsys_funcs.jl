using DataFrames
using CSV
using HTTP
using Printf
using LinearAlgebra
using Statistics
using Random
using JLD2
using FileIO
using StatsBase

"""

returns a fold assignment matrix, the same shape as `H`, where entry (i, j) is the fold to which the data point belongs. if a value is missing, we assign fold 0.
"""
function k_fold_split(H::Array{Union{Float64, Missing}, 2}, n_folds::Int, max_iter::Int=1000)
    @assert n_folds > 1 "Choose a `n_folds` greater than 1, otherwise this function is pointless."
    fold_assignment_matrix = fill(0, size(H))
    all_k_folds_not_represented_in_each_column = true
    n_iter = 0
    while all_k_folds_not_represented_in_each_column
        n_iter += 1
        all_k_folds_not_represented_in_each_column = false
        for mof_id = 1:size(H)[1]
            # grabs indices of gases in the current row (representing a MOF) that are not `missing`
            gas_ids = findall(! ismissing, H[mof_id, :])
            while length(gas_ids) >= n_folds
                # assign a random data point to each fold
                # this ensures that each fold has at least one data pt.
                # we keep doing this until length(gas_ids) < n_folds to ensure even spread among the folds
                for fold = 1:n_folds
                    # randomly choose a gas_id to assign to this fold
                    gas_id = sample(gas_ids)
                    fold_assignment_matrix[mof_id, gas_id] = fold
                    # remove this gas id from the list since it has been assigned
                    filter!(g -> g ≠ gas_id, gas_ids)
                end
            end
            # If there are leftover data points (i.e. #data points left < `n_folds`), we will randomly assign them into folds
            for gas_id in gas_ids
                fold_assignment_matrix[mof_id, gas_id] = rand(1:n_folds)
            end
        end
        # Now we have to make sure we have enough data points in the columns as well
        for gas_id = 1:size(H)[2]
            unique_folds = unique(fold_assignment_matrix[:, gas_id])
            # if not all folds are represented in this column, then make the boolean true
            #  so we can start over
            if length(unique_folds) <= n_folds
                all_k_folds_not_represented_in_each_column = true
                break
            end
        end
        # If too many iterations are performed, an Error is raised
        if n_iter > max_iter
            error("Maximum number of iterations reached. Try changing the `max_iter` argument.")
        end
    end
    # some sanity tests
    @assert all(ismissing.(H) .== (fold_assignment_matrix .== 0)) # missing values in 0 fold
    # each row has all n_folds
    for mof_id = 1:size(H)[1]
        @assert length(unique(fold_assignment_matrix[mof_id, :])) == n_folds + 1
    end

    @printf("Number of iterations requried to split data into %d-folds: %d\n", n_folds, n_iter)
    return fold_assignment_matrix
end

function ALS(H::Array{Union{Float64, Missing}, 2}, r::Int, λ::Array{Float64, 1};
             relative_loss_threshold::Float64=1e-5, als_sweeps_after_loss_stops_decreasing::Int=15, 
             max_als_sweeps::Int=20000, verbose::Bool=true)
    @assert length(λ) == 2 "There should be two λ values, one for each latent matrix."
    nb_als_sweeps = 0
    train_rmses = Array{Float64, 1}()
    train_rmse = Inf
    prev_train_rmse = 0.0
    losses = Array{Float64, 1}()
    loss = Inf
    prev_loss = 0.0

    nb_als_sweeps_after_loss_stops_decreasing = 0
    n_m = size(H)[1]
    n_g = size(H)[2]
    
    # pre-allocate latent reps and biases
    M = rand(r, n_m) .- 0.5
    G = rand(r, n_g) .- 0.5
    μ = rand(1, n_m) .- 0.5
    γ = rand(1, n_g) .- 0.5
    h̄ = mean(H[.!ismissing.(H)])
    
    # pre-allocate identity matrices for invoking regularization
    Im = λ[1] * Array{Float64, 2}(I, r, r) / n_m
    Ig = λ[1] * Array{Float64, 2}(I, r, r) / n_g
    bias_λ = [λ[2] / n_m, λ[2] / n_g]

    if verbose
        @printf("M shape: (%d, %d)\tG shape: (%d, %d)\n", size(M)[1], size(M)[2], size(G)[1], size(G)[2])
    end
    
    # to facilitate the random choice of a latent vector for a MOF and for a gas
    latent_vectors_to_update = vcat([(m, :mof) for m = 1:n_m], [(g, :gas) for g = 1:n_g])
    # Boolean array of which entries in H are not missing
    idx_H_nonmissing = .! ismissing.(H)
    nb_nonmissing = sum(idx_H_nonmissing)

    keep_doing_als_sweeps = true
    while keep_doing_als_sweeps
        # shuffle order for ALS
        shuffle!(latent_vectors_to_update)
        for latent_vector in latent_vectors_to_update
            if latent_vector[2] == :mof
                # index of MOF whose latent rep and bias we update
                mof_id = latent_vector[1]
                # get non-missing gas ids
                gas_ids = findall(! ismissing, H[mof_id, :])
                # update latent vector
                biased_H = H[mof_id, gas_ids] .- (h̄ .+ γ[1, gas_ids] .+ μ[1, mof_id])
                b = G[:, gas_ids] * biased_H
                A = G[:, gas_ids] * G[:, gas_ids]' + Im
                M[:, mof_id] = A \ b
                # update bias
                μ[1, mof_id] = 0.0
                for gas_id in gas_ids
                    μ[1, mof_id] += H[mof_id, gas_id] - γ[1, gas_id] - h̄ - M[:, mof_id]' * G[:, gas_id]
                end
                μ[1, mof_id] /= length(gas_ids) + bias_λ[1]
            else
                # index of gas whose latent rep and bias we update
                gas_id = latent_vector[1]
                # get non-missing mof ids
                mof_ids = findall(! ismissing, H[:, gas_id])
                # update latent vector
                biased_H = H[mof_ids, gas_id] .- (h̄ .+ μ[1, mof_ids] .+ γ[1, gas_id])
                b = M[:, mof_ids] * biased_H
                A = M[:, mof_ids] * M[:, mof_ids]' + Ig
                G[:, gas_id] = A \ b
                # update bias
                γ[1, gas_id] = 0.0
                for mof_id in mof_ids
                    γ[1, gas_id] += H[mof_id, gas_id] - μ[1, mof_id] - h̄ - M[:, mof_id]' * G[:, gas_id]
                end
                γ[1, gas_id] /= length(mof_ids) + bias_λ[2]
            end
        end
    
        # compute sum of square errors on training data
        h_predicted = M' * G .+ h̄ .+ μ' .+ γ
        sse = sum((H[idx_H_nonmissing] - h_predicted[idx_H_nonmissing]) .^ 2)
    
        # update RMSE on training data
        prev_train_rmse = train_rmse
        train_rmse = sqrt(sse / nb_nonmissing)
        push!(train_rmses, train_rmse)
        rmse_diff = abs(prev_train_rmse - train_rmse)
        
        # update loss
        prev_loss = loss
        loss = 0.5 * sse # error term
        loss += 0.5 * λ[1] * (sum([norm(M[:, m])^2 for m = 1:n_m]) / n_m + sum([norm(G[:,g]) for g = 1:n_g]) / n_g)
        loss += 0.5 * λ[2] * (norm(μ)^2 / n_m + norm(γ)^2 / n_g)
        push!(losses, loss)
        loss_diff = prev_loss - loss

        # print info every sweep
        if nb_als_sweeps % 10 == 0 && verbose
            println("ALS sweep ", nb_als_sweeps)
            println("\ttraining loss = ", loss)
            println("\ttraining RMSE = ", train_rmse)
        end
        
        ###
        ### stopping criteria
        ###
        # stop if we've reached the maximum number of ALS sweeps
        nb_als_sweeps += 1
        if nb_als_sweeps > max_als_sweeps
            @printf("Maximum number of ALS sweeps (%d) reached.\n", max_als_sweeps)
            keep_doing_als_sweeps = false
        end
        
        # stop if the loss stopped decreasing for a certain number of iterations
        rel_loss_change = (prev_loss - loss) / loss # greater than zero if loss decreased
        if abs(rel_loss_change) < relative_loss_threshold
            nb_als_sweeps_after_loss_stops_decreasing += 1
            if nb_als_sweeps_after_loss_stops_decreasing >= als_sweeps_after_loss_stops_decreasing
                println("loss stopped decreasing")
                keep_doing_als_sweeps = false
            end
        end
    end

    if any(isnan.(M)) || any(isnan.(G)) || any(ismissing.(M)) || any(ismissing.(G))
        error("NaN or missing encountered in either latent representation")
        return M, G
    end

    return M, G, μ, γ, h̄, train_rmses, losses
end

function cross_validation(H::Array{Union{Float64, Missing}, 2}, fold_matrix::Array{Int, 2}, r::Int, λ₁::Float64, λ₂::Float64)
    k = length(unique(fold_matrix)) - 1
    @printf("------------------------------\nStarting ALS with the following parameters:\n\tr = %d, λ = [%.4f, %.4f]\n", r, λ₁, λ₂)
    test_errors = Array{Float64, 1}()
    parity_prediction = zeros(size(H))

    for test_fold = 1:k
        M, G, train_error = ALS(replace(H .* (fold_matrix .!= test_fold), 0.0=>missing), r, [λ₁, λ₂], 1e-6, 1e-7, 10000, false)
        pred = (M' * G)[fold_matrix .== test_fold]
        parity_prediction[fold_matrix .== test_fold] = pred
        actual = H[fold_matrix .== test_fold]
        append!(test_errors, sqrt(sum((actual - pred).^2)/length(pred)))
    end
    test_error = mean(test_errors)
    @printf("Test Error: %.4f\n", test_error)
    return test_error, parity_prediction
end


function write_submit_file(H::Array{Union{Float64, Missing}, 2}, r::Int, λ₁::Float64, λ₂::Float64, idx1::Int, idx2::Int)
    jobscriptdir = "jobz"
    if ! isdir(jobscriptdir)
        mkdir(jobscriptdir)
    end
	filename = @sprintf("LOO_%d_%.3f_%.3f_%d_%d", r, λ₁, λ₂, idx1, idx2)
#	@printf("Running the following job: %s\n", filename)

    submit_file = open(filename * ".sh", "w")
    @printf(submit_file,
    """
    #!/bin/bash

    # use current working directory for input and output
    # default is to use the users home directory
    #\$ -cwd

    # name of this job
    #\$ -N %s

    #\$ -pe thread 1 # use 1 thread/core per job
    
    # send stderror to this file
    #\$ -e jobz/%s.e
    #\$ -o jobz/%s.o

    /nfs/stak/users/sturlusa/julia-1.1.0/bin/julia run_loo.jl %d %f %f %d %d
    """,
    filename, filename, filename, r, λ₁, λ₂, idx1, idx2)
    close(submit_file)
    return filename * ".sh"
end

function concatenate_data(file_root::AbstractString, H_shape::Tuple{Int, Int})
	correct_files = [file for file in readdir("results") if occursin(file_root, file)]
	n = length(correct_files)
	parity_pred = Array{Union{Float64, Missing}, 2}(undef, H_shape[1], H_shape[2])
	err = Array{Float64, 1}(undef, n)
	loss = Array{Float64, 1}(undef, n)

	for (i, file) in enumerate(correct_files)
		data = CSV.read("results/" * file)
		idx1 = parse(Int, split(split(file, ".csv")[1], "_")[end-1])
		idx2 = parse(Int, split(split(file, ".csv")[1], "_")[end])
		err[i] = data[!, :err][1]	
		loss[i] = data[!, :loss][1]	
		parity_pred[idx1, idx2] = data[!, :parpred][1]
	end
	output_filename = file_root * ".jld2"
	save(output_filename, Dict("err" => err, "loss" => loss, "paritypred" => parity_pred))
	for file in correct_files
		run(`rm -f results/$file`)
		outfile = split(file, ".csv")[1] * ".o"
		errfile = split(file, ".csv")[1] * ".e"
		run(`rm -f jobz/$outfile`)
		run(`rm -f jobz/$errfile`)
	end
	return
end

function LOO_for_cluster(H::Array{Union{Float64, Missing}, 2}, r::Int, λ₁::Float64, λ₂::Float64)
    non_missing_indices = findall(.!ismissing.(H))
	n = length(non_missing_indices)

    for (i, index) in enumerate(non_missing_indices)
		filename = @sprintf("LOO_%d_%.3f_%.3f_%d_%d", r, λ₁, λ₂, index[1], index[2])
		if any(occursin.(filename, readdir("results")))	|| any(occursin.(@sprintf("LOO_%d_%.3f_%.3f.jld2", r, λ₁, λ₂), readdir()))
			continue
		end
		submit_file_name = write_submit_file(H, r, λ₁, λ₂, index[1], index[2])
        run(`qsub $submit_file_name`)
        sleep(0.25)
		run(`rm -f $submit_file_name`)
    end
	file_root = @sprintf("LOO_%d_%.3f_%.3f_", r, λ₁, λ₂)

	n_iter = 0
	while true
		n_iter += 1
		n_files = 0
		result_files = readdir("results")
		if sum(occursin.(file_root, result_files)) == n
			concatenate_data(file_root, size(H))
			break
		else
			if n_iter > 10
				break
			end
			sleep(60)
		end
	end
    return
end

function LOO_cross_validation(H::Array{Union{Float64, Missing}, 2}, r::Int, λ₁::Float64, λ₂::Float64)
    @printf("------------------------------\nStarting LOO-ALS with the following parameters:\n\tr = %d, λ = [%.4f, %.4f]\n", r, λ₁, λ₂)
    non_missing_indices = findall(.!ismissing.(H))
    n = length(non_missing_indices)
    test_errors = Array{Float64, 1}(undef, n)
    parity_prediction = zeros(size(H))

    for (i, index) in enumerate(non_missing_indices)
        temp_H = copy(H)
        temp_H[index] = missing
        M, G, mu, gamma, train_error, train_loss, hbar = ALS(temp_H, r, [λ₁, λ₂], 1e-6, 1e-7, 100, false)
        pred = M[:, index[1]]' * G[:, index[2]] + mu[1, index[1]] + gamma[1, index[2]] + hbar
        parity_prediction[index] = pred
        actual = H[index]
        test_errors[i] = sqrt((actual - pred)^2)
    end
    test_error = mean(test_errors)
    @printf("Test Error: %.4f\n", test_error)
    return test_error, parity_prediction
end

function mean_cross_validation(H::Array{Union{Float64, Missing}, 2}, fold_matrix::Array{Int, 2}, axis::AbstractString)
    axis = lowercase(axis)
    @assert lowercase(axis) in ["gas", "mof"]
    mean_parity_pred = zeros(size(H))
    k = length(unique(fold_matrix)) - 1
    dim = Int(axis != "mof") + 1

    for test_fold = 1:k
        train_mean = missing_mean(replace(H .* (fold_matrix .!= test_fold), 0.0=>missing), dim)
        for row = 1:size(H)[1]
            for col = 1:size(H)[2]
                if fold_matrix[row,col] == test_fold
                    if dim == 1
                        mean_parity_pred[row, col] = train_mean[row]
                    else
                        mean_parity_pred[row, col] = train_mean[col]
                    end
                end
            end
        end
    end
    return mean_parity_pred
end


function missing_mean(H::Array{Union{Missing, Float64},2}, axis::Int)
    @assert axis in [1, 2]
    means = zeros(size(H)[axis])
    for i = 1:size(H)[axis]
        if axis == 1
            means[i] = mean(skipmissing(H[i,:]))
        else
            means[i] = mean(skipmissing(H[:,i]))
        end
    end
    return means
end

