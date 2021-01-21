### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 92083d94-5b82-11eb-2274-ed62139bbf2d
begin
	using LowRankModels, CSV, DataFrames, PyPlot, Statistics, Distributions, StatsBase
	using ScikitLearn.CrossValidation: train_test_split
end

# ╔═╡ ae415fa4-5b82-11eb-0051-072097bb0439
md"# read in data"

# ╔═╡ b3deec9e-5b82-11eb-0e37-abd2ac9d4b44
begin
	df = CSV.read("aiida_ads_data_oct20.csv", DataFrame)
	
	# log-10 transform the henry coefficients.
	for henry_col in Symbol.(["h2o_henry", "h2s_henry", "xe_henry", "kr_henry"])
	    # convert units from mol/(kg-Pa) to mol/(kg-bar)
	    df[!, henry_col] = 1e5 * df[:, henry_col]
	    # log-10 transform
	    df[!, henry_col] = log10.(df[:, henry_col])
	end
	
	head(df)
end

# ╔═╡ bf9ed538-5b82-11eb-1198-3d35a209c5c0
md"# construct material-property matrix"

# ╔═╡ c5d42b2e-5b82-11eb-0631-35efb6b0800c
begin
	properties = ["o2_5bar", 
	              "o2_140bar", 
	              "co2_0.001bar", 
	              "co2_30bar", 
	              "n2_0.001bar", 
	              "n2_30bar", 
	              "h2_77K_5bar", 
	              "h2_77K_100bar", 
	              "h2_298K_5bar",
	              "h2_298K_100bar", 
	              "h2o_henry", 
	              "h2s_henry",
	              "xe_henry",
	              "kr_henry",
	              "ch4_65bar",
	              "ch4_5.8bar"
	              ]
	
	prop_to_label = Dict("o2_5bar"        => L"O$_2$" * "\n(298 K, 5 bar)", 
	                     "o2_140bar"      => L"O$_2$" * "\n(298 K, 140 bar)", 
	                     "co2_0.001bar"   => L"CO$_2$" * "\n(300 K, 0.001 bar)", 
	                     "co2_30bar"      => L"CO$_2$" * "\n(300 K, 30 bar)", 
	                     "n2_0.001bar"    => L"N$_2$" * "\n(300 K, 0.001 bar)", 
	                     "n2_30bar"       => L"N$_2$" * "\n(300 K, 30 bar)", 
	                     "h2_77K_5bar"    => L"H$_2$" * "\n(77 K, 5 bar)", 
	                     "h2_77K_100bar"  => L"H$_2$" * "\n(77 K, 100 bar)", 
	                     "h2_298K_5bar"   => L"H$_2$" * "\n(298 K, 5 bar)", 
	                     "h2_298K_100bar" => L"H$_2$" * "\n(298 K, 100 bar)", 
	                     "h2o_henry"      => L"H$_2$O Henry" * "\n(300 K)", 
	                     "h2s_henry"      => L"H$_2$S" * "\nHenry Coeff (300 K)", 
	                     "xe_henry"       => "Xe Henry\n (300 K)", 
	                     "kr_henry"       => "Kr Henry\n (300 K)", 
	                     "ch4_65bar"      => L"CH$_4$" * "\n(298 K, 65 bar)", 
	                     "ch4_5.8bar"     => L"CH$_4$" * "\n(298 K, 5.8 bar)"
	                    )

	const materials = df[:, :cof]
	const n_m = length(materials)
	const n_p = length(properties)
	# material property matrix
	const A_complete_unnormalized = convert(Matrix, df[:, properties])
end

# ╔═╡ c4e8d0b0-5b83-11eb-1cb3-8125c5d3c4ae
begin
	function viz_scatterplot_matrix()
		# normalize columns
		A_n = deepcopy(A_complete_unnormalized)
		for i = 1:n_p
			A_n[:, i] = (A_n[:, i] .- mean(A_n[:, i])) ./ std(A_n[:, i])
		end
		
		fig, axs = subplots(n_p, n_p, figsize=(30, 30))
		hist_ylims = (0, 600)
		for i = 1:n_p
			for j = 1:n_p
				property_i = prop_to_label[properties[i]]
				property_j = prop_to_label[properties[j]]
				if j > i
					axs[i, j].axis("off")
					continue
				end
				
				# histogram
				if i == j
					axs[i, j].set_ylim(hist_ylims)
					axs[i, j].hist(A_n[:, i], 
						           ec="k", fc="#fabebe", alpha=0.7)
					if i == 1
						axs[i, j].set_ylabel("# COFs")
					else
						axs[i, j].spines["left"].set_visible(false)
						axs[i, j].set_yticks([])
					end
						
					if j == n_p
						axs[i, j].set_xlabel(property_i)
					else
						axs[i, j].set_xticks([])
					end
				# scatter plot
				else
					axs[i, j].scatter(A_n[:, j], A_n[:, i], 
						              ec="k", fc="#fabebe", alpha=0.7)
					if i == n_p
						axs[i, j].set_xlabel(property_j)
					else
						axs[i, j].set_xticks([])
					end
					if j == 1
						axs[i, j].set_ylabel(property_i)
					else
						axs[i, j].set_yticks([])
					end
				end
				axs[i, j].spines["top"].set_visible(false)
				axs[i, j].spines["right"].set_visible(false)
			end
		end
		# tight_layout()
		savefig("scatterplot_matrix.png", dpi=300, format="png")
		gcf()
	end
	
	viz_scatterplot_matrix()
end

# ╔═╡ 6713990c-5b8d-11eb-2196-7182f36cad59
md"# simulate data collection"

# ╔═╡ 2931005c-5b8d-11eb-2375-5dacf441be72
# θ: target fraction observed values
function sim_data_collection(θ::Float64)
    distn = Bernoulli(θ) # for observing it
    A = zeros(Union{Float64, Missing}, n_m, n_p)
	for m = 1:n_m
		for p = 1:n_p
			if rand(distn)
				A[m, p] = A_complete_unnormalized[m, p]
			else
				A[m, p] = missing
			end
		end
	end
	θ_true = 1.0 - sum(ismissing.(A)) / (n_m * n_p)
    return A, θ_true
end

# ╔═╡ a21ac3b8-5ba1-11eb-2d70-bdf4b395f563
function ids_obs_and_unobs(A::Array{Union{Float64, Missing}, 2})
	ids_obs = observations(A) # returns tuple of observations
	ids_not_obs = [(i, j) for i=1:size(A)[1], j=1:size(A)[2] if !((i,j) in ids_obs)][:]
	return ids_obs, ids_not_obs
end

# ╔═╡ 1d363324-5b8f-11eb-2c78-6980a0d5f110
md"# normalization of columns"

# ╔═╡ 2366b3cc-5b8f-11eb-07e4-61cbc97d5480
function normalize!(A::Array{Union{Float64, Missing}, 2})
	μs = zeros(n_p)
	σs = zeros(n_p)
    for p = 1:n_p
        ids_obs = .! ismissing.(A[:, p])
		μs[p] = mean(A[ids_obs, p])
		σs[p] = std(A[ids_obs, p])
        A[:, p] .= (A[:, p] .- μs[p]) ./ σs[p]
    end
	return μs, σs
end

# ╔═╡ 6ef474cc-5b90-11eb-2fe2-87bc7b48f5b7
md"# fit low rank model"

# ╔═╡ 8a3c55ae-5ba4-11eb-3354-f9a8feaa7e91
struct HyperParam
	k::Int
	λ::Float64
end

# ╔═╡ 1d745ea2-5ba5-11eb-274a-4ff7a65b0cb6
struct ValidationRun
	hyper_param::HyperParam
	rmsd::Float64
end

# ╔═╡ cc771288-5ba6-11eb-1792-9dfc54c58a8c
function min_rmsd(vrs::Array{ValidationRun, 1})
	da_best = vrs[1]
	for i = 2:length(vrs)
		if vrs[i].rmsd < da_best.rmsd
			da_best = vrs[i]
		end
	end
	return da_best
end

# ╔═╡ f7158eea-5ba4-11eb-17cc-d57c477cec39
function hp_grid(ks::Array{Int, 1}, λs::Array{Float64, 1})
	hps = HyperParam[]
	for k in ks
		for λ in λs
			push!(hps, HyperParam(k, λ))
		end
	end
	return hps
end

# ╔═╡ 8152d710-5b90-11eb-39f5-45d81aa298ab
# k = rank of matrix
# λ = regularization param
# obs = which observations we train on.
# gotta pass the transpose for the bias to work.
function fit_glrm(At::Array{Union{Float64, Missing}, 2}, 
		          hp::HyperParam,
		          obs::Array{Tuple{Int64,Int64}, 1})
	# quadratic regularizors
    rp = QuadReg(hp.λ)
    rm = QuadReg(hp.λ * n_p / n_m)
	# this should be the transpose...
	@assert size(At) == (n_p, n_m)
	
	# A = M' P
	# At = P' M
	#   so here X is the analogy of P; Y is the analogy of M
    glrm = GLRM(At, QuadLoss(), rp, rm, hp.k, obs=obs, offset=true)
#    init_svd!(glrm)
    P, M, ch = fit!(glrm)
    @assert size(P) == (hp.k, n_p)
    @assert size(M) == (hp.k, n_m)

    @assert isapprox(impute(glrm), P' * M)
    return P, M, glrm, ch
end

# ╔═╡ e2dd554c-5baf-11eb-1b49-654d19bedecc
md"# for evaluating the glrm"

# ╔═╡ 168d4c44-5bb3-11eb-2d16-af69f613b625
# get the true, normalized matrix
function compute_At_complete(μs::Array{Float64, 1}, σs::Array{Float64, 1})
	An = deepcopy(A_complete_unnormalized)
	for p = 1:n_p
        An[:, p] .= (An[:, p] .- μs[p]) ./ σs[p]
    end
	return collect(An')
end

# ╔═╡ 36da7c1e-5bb1-11eb-2bc9-438dabcf9cc5
# spearman rank for property p
#    true A': true matrix
#    pred A': imputed matrix
#    ids_test: the test ids
#    p: the property
function ρ_p(𝓅::Int,
		     At_complete::Array{Float64, 2},
			 Ât::Array{Number, 2},
		     ids_test::Array{Tuple{Int64,Int64}, 1}
		    )
	@assert size(Ât) == (n_p, n_m)
	
	ids_test_𝓅 = [(p, m) for (p, m) in ids_test if p == 𝓅]
	
	# truth and predicted value of the property
	a = [At_complete[p, m] for (p, m) in ids_test_𝓅]
	â = convert(Array{Float64, 1}, 
		        [Ât[p, m] for (p, m) in ids_test_𝓅]
		        )
	return corspearman(a, â)
end

# ╔═╡ efc74f24-5ba0-11eb-2c44-6dac87ec534a
md"# hyperparam grid sweep"

# ╔═╡ a269ab26-5ba4-11eb-1001-6703f57f495c
begin
	ks = collect(2:3)                       # ranks
	λs = 10.0 .^ range(-1.0, 3.0, length=3) # regularization params
	hyper_params = hp_grid(ks, λs)
end

# ╔═╡ 9cf75472-5ba0-11eb-371a-5bc338946b61
# hyperparam sweep
# return optimal hyperparams with min rmsd
function hyperparam_sweep(hyper_params::Array{HyperParam, 1}, 
						  At::Array{Union{Float64, Missing}, 2},
						  ids_train::Array{Tuple{Int64,Int64},1},
						  ids_valid::Array{Tuple{Int64,Int64},1}
						 ) 
	valid_runs = ValidationRun[]
	for hyper_param in hyper_params
		# train on training data
		G, M, glrm, ch = fit_glrm(At, hyper_param, ids_train)
		# impute missing entries
		Ât = impute(glrm)
		# evaluate on validation data
		a = [At[p, m] for (p, m) in ids_valid]
		â = [Ât[p, m] for (p, m) in ids_valid]

		push!(valid_runs, ValidationRun(hyper_param, 
										rmsd(a, â)
									   )
			  )
	end
	
	return min_rmsd(valid_runs)
end

# ╔═╡ 2009751a-5bae-11eb-158f-a3d9cb98fe24
md"# performance evaluation"

# ╔═╡ 09b74408-5baa-11eb-3735-e9333756f565
struct TestResult
	# gas-wise Spearman rank coeffs
	ρs::Array{Float64, 1}
	# true test data pts
	a::Array{Float64, 1}
	# pred test data pts
	â::Array{Float64, 1}
end

# ╔═╡ b285e2fe-5ba7-11eb-2e12-83e72bcafa2f
begin
	# 1. simulate data collection
	# 2. split observed values into train and validation set
	# 3. remaining unobserved = test data
	# 4. do hyperparam sweep to find optimal hyperparams, judged by validation data
	# 5. now that we hv optimal hyper params, retrain on all observed.
	# 6. evaluate it on test entries
	#   return (true values, pred values)
	function run_simulation(θ::Float64)
		A, θ_true = sim_data_collection(θ)
		
		# store for later.
		μs, σs = normalize!(A)

		At = collect(A')

		ids_obs, ids_unobs = ids_obs_and_unobs(At)
		ids_test = ids_unobs
		@assert(all(ismissing.([At[p, m] for (p, m) in ids_test])))
		ids_train, ids_valid = train_test_split(ids_obs, test_size=0.2, shuffle=true)

		# hyper-parameter sweep to find optimal hyper param
		opt_valid_run = hyperparam_sweep(hyper_params, At, ids_train, ids_valid)

		# train model on all observed
		G, M, glrm, ch = fit_glrm(At, opt_valid_run.hyper_param, ids_obs)
		
		# impute unobserved entries
		Ât = impute(glrm)
		
		# compute truth from normalizations
		At_complete = compute_At_complete(μs, σs)
		
		# test model on unobserved
		a = [At_complete[p, m] for (p, m) in ids_test]
		â = [Ât[p, m] for (p, m) in ids_test]

		# get spearman ranks for gases
		
		ρs = [ρ_p(p, At_complete, Ât, ids_test) for p = 1:n_p]
		return TestResult(ρs, a, â)
	end
	
	θ = 0.6
	test_result = run_simulation(θ)
end

# ╔═╡ a6b959aa-5bb2-11eb-3ab4-eb90f14f6f61
begin
	figure()
	scatter(test_result.a, test_result.â)
	gcf()
end

# ╔═╡ Cell order:
# ╠═92083d94-5b82-11eb-2274-ed62139bbf2d
# ╟─ae415fa4-5b82-11eb-0051-072097bb0439
# ╠═b3deec9e-5b82-11eb-0e37-abd2ac9d4b44
# ╟─bf9ed538-5b82-11eb-1198-3d35a209c5c0
# ╠═c5d42b2e-5b82-11eb-0631-35efb6b0800c
# ╠═c4e8d0b0-5b83-11eb-1cb3-8125c5d3c4ae
# ╟─6713990c-5b8d-11eb-2196-7182f36cad59
# ╠═2931005c-5b8d-11eb-2375-5dacf441be72
# ╠═a21ac3b8-5ba1-11eb-2d70-bdf4b395f563
# ╟─1d363324-5b8f-11eb-2c78-6980a0d5f110
# ╠═2366b3cc-5b8f-11eb-07e4-61cbc97d5480
# ╟─6ef474cc-5b90-11eb-2fe2-87bc7b48f5b7
# ╠═8a3c55ae-5ba4-11eb-3354-f9a8feaa7e91
# ╠═1d745ea2-5ba5-11eb-274a-4ff7a65b0cb6
# ╠═cc771288-5ba6-11eb-1792-9dfc54c58a8c
# ╠═f7158eea-5ba4-11eb-17cc-d57c477cec39
# ╠═8152d710-5b90-11eb-39f5-45d81aa298ab
# ╟─e2dd554c-5baf-11eb-1b49-654d19bedecc
# ╠═168d4c44-5bb3-11eb-2d16-af69f613b625
# ╠═36da7c1e-5bb1-11eb-2bc9-438dabcf9cc5
# ╟─efc74f24-5ba0-11eb-2c44-6dac87ec534a
# ╠═a269ab26-5ba4-11eb-1001-6703f57f495c
# ╠═9cf75472-5ba0-11eb-371a-5bc338946b61
# ╟─2009751a-5bae-11eb-158f-a3d9cb98fe24
# ╠═09b74408-5baa-11eb-3735-e9333756f565
# ╠═b285e2fe-5ba7-11eb-2e12-83e72bcafa2f
# ╠═a6b959aa-5bb2-11eb-3ab4-eb90f14f6f61
