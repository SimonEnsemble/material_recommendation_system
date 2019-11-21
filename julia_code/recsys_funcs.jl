using DataFrames
using CSV
using HTTP
using Printf
using LinearAlgebra
using Statistics
using Random
using JLD2
using FileIO

function k_fold_split(H::Array{Union{Float64, Missing}, 2}, k::Int, max_iter::Int=1000)
    @assert k > 1 "Choose a `k` greater than 1, otherwise this function is pointless."
    fold_matrix = fill(0, size(H))
    all_k_folds_not_represented_in_each_column = true
    n_iter = 0
    while all_k_folds_not_represented_in_each_column
        n_iter += 1
        all_k_folds_not_represented_in_each_column = false
        for i_row = 1:size(H)[1]
            # Grabs available indices (the ones that are not `missing`
            avail_indices = findall(x->typeof(x) != Missing, H[i_row,:])
            # As long as there are enough data points to distribute between `k` folds, we will equally distribute them
            while length(avail_indices) >= k
                for fold = 1:k
                    j_col = rand(collect(1:length(avail_indices)))
                    fold_matrix[i_row, avail_indices[j_col]] = fold
                    filter!(e->e≠avail_indices[j_col], avail_indices)
                end
            end
            # If there are leftover data points (i.e. #data points left < `k`), we will randomly assign them into folds
            for j_col in avail_indices
                fold_matrix[i_row, j_col] = rand(collect(1:k))
            end
        end
        # Now we have to make sure we have enough data points in the columns as well
        for j_col = 1:size(H)[2]
            unique_folds = unique(fold_matrix[:, j_col])
            if length(unique_folds) <= k
                all_k_folds_not_represented_in_each_column = true
                break
            end
        end
        # If too many iterations are performed, an Error is raised
        if n_iter > max_iter
            error("Maximum number of iterations reached. Try changing the `max_iter` argument.")
        end
    end
    @printf("Number of iterations requried to split data into %d-folds: %d\n", k, n_iter)
    return fold_matrix
end

function ALS(H::Array{Union{Float64, Missing}, 2}, r::Int, λ::Array{Float64, 1}, error_threshold::Float64, convergence_threshold::Float64=1e-6, maxiter::Int=20000, verbose::Bool=true)
    @assert length(λ) == 2 "There should be two λ values, one for each latent matrix."
    n = 0
    train_error_array = Array{Float64, 1}()
    train_error = Inf
    prev_error = 0.0
    loss_arr = Array{Float64, 1}()
    loss = Inf
    prev_loss = 0.0

    convergence_count = 0
    n_m = size(H)[1]
    n_g = size(H)[2]
    H_missing_mask = .!ismissing.(H)

    M = rand(r, n_m) .- 0.5
    G = rand(r, n_g) .- 0.5
    mu = rand(1, n_m) .- 0.5
    gamma = rand(1, n_g) .- 0.5
    hbar = mean(H[.!ismissing.(H)])
    Im = n_g / n_m * λ[1] * Array{Float64, 2}(I, r, r)
    Ig = λ[1] * Array{Float64, 2}(I, r, r)
    bias_λ = [n_g / n_m * λ[2], λ[2]]

    if verbose
        @printf("M shape: (%d, %d)\tG shape: (%d, %d)\n", size(M)[1], size(M)[2], size(G)[1], size(G)[2])
    end

    vector_to_pick = vcat([(m, "m") for m = 1:n_m], [(g, "g") for g = 1:n_g])
    while loss > error_threshold
        shuffle!(vector_to_pick)
        for vector in vector_to_pick
            if vector[2] == "m"
                m = vector[1] 
                gases_in_which_H_of_this_mof_is_measured = .!ismissing.(H[m,:])
                biased_H = H[m, gases_in_which_H_of_this_mof_is_measured] .- (hbar .+ gamma[1, gases_in_which_H_of_this_mof_is_measured] .+ mu[1, m])
                b = G[:, gases_in_which_H_of_this_mof_is_measured] * biased_H
                A = G[:, gases_in_which_H_of_this_mof_is_measured] * G[:, gases_in_which_H_of_this_mof_is_measured]' + Im
                M[:, m] = A\b
                mu[1, m] = sum([H[m, g] - gamma[1, g] - hbar - M[:, m]' * G[:, g] for g = 1:n_g][gases_in_which_H_of_this_mof_is_measured]) / (sum(gases_in_which_H_of_this_mof_is_measured) + bias_λ[1])
            else
                g = vector[1]
                mofs_in_which_H_of_this_gas_is_measured = .!ismissing.(H[:,g])
                biased_H = H[mofs_in_which_H_of_this_gas_is_measured, g] .- (hbar .+ mu[1, mofs_in_which_H_of_this_gas_is_measured] .+ gamma[1, g])
                b = M[:, mofs_in_which_H_of_this_gas_is_measured] * biased_H
                A = M[:, mofs_in_which_H_of_this_gas_is_measured] * M[:, mofs_in_which_H_of_this_gas_is_measured]' + Ig
                G[:, g] = A\b
                gamma[1, g] = sum([H[m, g] - mu[1, m] - hbar - M[:, m]' * G[:, g] for m = 1:n_m][mofs_in_which_H_of_this_gas_is_measured]) / (sum(mofs_in_which_H_of_this_gas_is_measured) + bias_λ[2])
            end
        end

        pred = (M' * G .+ hbar .+ mu' .+ gamma)[H_missing_mask][:]
        actual = H[H_missing_mask][:]
        prev_error = train_error
        train_error = sqrt(sum((actual - pred).^2)/length(pred))
        append!(train_error_array, train_error)
        error_diff = abs(prev_error - train_error)
        prev_loss = loss
        loss = 0.5 * sum((actual-pred).^2) + 0.5 * λ[1] * (n_g / n_m * sum([norm(M[:,m])^2 for m = 1:n_m]) + sum([norm(G[:,g]) for g = 1:n_g])) + 0.5 * λ[2] * (n_g / n_m * norm(mu)^2 + norm(gamma)^2)
        append!(loss_arr, loss)
        loss_diff = prev_loss - loss
        if n % 1000 == 0 && verbose
            @printf("Train loss on iteration %d: %.3f\n---------------\n", n, loss)
        end

        if error_diff < convergence_threshold
            convergence_count += 1
            if convergence_count > 199
                if verbose
                    @printf("Training has converged after %d iterations. See `convergence` parameter for convergence procedure.\n", n)
                    @printf("Train loss: %.3f\n", loss)
                end
                break
            end
        else
            convergence_count = 0
        end
        n += 1
        if n > maxiter
            @printf("Maximum number of iterations (%d) reached.\n", maxiter)
            break
        end
        if sum(skipmissing(isnan.(M))) > 0 || sum(skipmissing(isnan.(G))) > 0
            return M, G
            error("NaN encountered in either latent representation")
        end
    end

    if verbose
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(collect(1:length(loss_arr)), loss_arr, color="red")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Train Loss")
        plt.grid(true)
        plt.tight_layout()
        plt.savefig("asdf2.png", format="png", dpi=300)
    end
    return M, G, mu, gamma, train_error_array[end], loss_arr[end], hbar
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

