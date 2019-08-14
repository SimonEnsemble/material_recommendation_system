using DataFrames
using CSV
using HTTP
using Printf
using LinearAlgebra
using Statistics
using PyPlot


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

function predict(m::Array{Union{Float64, Missing}, 1}, g::Array{Union{Float64, Missing}, 1}, μ::Float64)
    return μ + dot(m, g)
end

function predict_all(M::Array{Union{Float64, Missing}, 2}, G::Array{Union{Float64, Missing}, 2}, μ::Float64)
    nm = size(M)[2]
    ng = size(G)[2]
    predictions = zeros(nm, ng)
    for m = 1:nm
        for g = 1:ng
            predictions[m, g] = predict(M[:,m], G[:,g], μ)
        end
    end
    return predictions
end

function ALS(H::Array{Union{Float64, Missing}, 2}, r::Int, λ::Array{Float64, 1}, error_threshold::Float64, convergence_threshold::Float64=1e-6, maxiter::Int=20000, verbose::Bool=true)
    @assert length(λ) == 2 "There should be two λ values, one for each latent matrix."
    n = 0
    train_error_array = Array{Float64, 1}()
    train_error = Inf
    prev_error = 0.0
    convergence_count = 0
    n_m = size(H)[1]
    n_g = size(H)[2]
    H_missing_mask = .!ismissing.(H)

    M = rand(r+2, n_m)
    G = rand(r+2, n_g)
    M[end-1, :] .= 1.0
    G[end, :] .= 1.0
    if verbose
        @printf("M shape: (%d, %d)\tG shape: (%d, %d)\n", size(M)[1], size(M)[2], size(G)[1], size(G)[2])
    end

    while train_error > error_threshold
        for m = 1:n_m
            missing_mask = .!ismissing.(H[m,:])
            b = G[:, missing_mask] * H[m, missing_mask]
            A = G[:, missing_mask] * G[:, missing_mask]' + λ[1] * Array{Float64, 2}(I, r+2, r+2)
            M[:, m] = A\b
            M[end-1, m] = 1.0
        end
        for g = 1:n_g
            missing_mask = .!ismissing.(H[:,g])
            b = M[:, missing_mask] * H[missing_mask, g]
            A = M[:, missing_mask] * M[:, missing_mask]' + λ[2] * Array{Float64, 2}(I, r+2, r+2)
            G[:, g] = A\b
            G[end, g] = 1.0
        end

        pred = (M' * G)[H_missing_mask][:]
        actual = H[H_missing_mask][:]
        prev_error = train_error
        train_error = sqrt(sum((actual - pred).^2)/length(pred))
        append!(train_error_array, train_error)
        error_diff = abs(prev_error - train_error)
        if n % 1000 == 0 && verbose
            @printf("Train error on iteration %d: %.3f\n---------------\n", n, train_error)
        end

        if error_diff < convergence_threshold
            convergence_count += 1
            if convergence_count > 199
                if verbose
                    @printf("Training has converged after %d iterations. See `convergence` parameter for convergence procedure.\n", n)
                    @printf("Train error: %.3f\n", train_error)
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
        ax.plot(collect(1:length(train_error_array)), train_error_array, color="red")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Train Loss")
        plt.grid(true)
        plt.tight_layout()
        plt.savefig("asdf.png", format="png", dpi=300)
    end
    return M, G, train_error_array[end]
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


function LOO_cross_validation(H::Array{Union{Float64, Missing}, 2}, r::Int, λ₁::Float64, λ₂::Float64)
    @printf("------------------------------\nStarting LOO-ALS with the following parameters:\n\tr = %d, λ = [%.4f, %.4f]\n", r, λ₁, λ₂)
    test_errors = Array{Float64, 1}()
    parity_prediction = zeros(size(H))
    non_missing_indices = findall(.!ismissing.(H))

    for index in non_missing_indices
        temp_H = copy(H)
        temp_H[index] = missing
        M, G, train_error = ALS(temp_H, r, [λ₁, λ₂], 1e-6, 1e-7, 10000, false)
        pred = M[:, index[1]]' * G[:, index[2]]
        parity_prediction[index] = pred
        actual = H[index]
        append!(test_errors, sqrt((actual - pred)^2))
    end
    test_error = mean(test_errors)
    @printf("Test Error: %.4f\n", test_error)
    return test_error, parity_prediction
end
