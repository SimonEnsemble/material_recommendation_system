### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 92083d94-5b82-11eb-2274-ed62139bbf2d
begin
	using LowRankModels, CSV, DataFrames, PyPlot, Statistics, Distributions, StatsBase, Printf, UMAP
	using ScikitLearn.CrossValidation: train_test_split
	PyPlot.matplotlib.style.use("https://gist.githubusercontent.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea/raw/64a78942d3f7b4b5054902f2cee84213eaff872f/matplotlibrc")
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

	const materials = map(x -> split(x, ".cif")[1], df[:, :cof])
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
						           ec="k", fc="C0", alpha=0.7)
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
					# axs[i, j].spines["left"].set_visible(true)
					# axs[i, j].spines["bottom"].set_visible(true)
					axs[i, j].scatter(A_n[:, j], A_n[:, i], 
						              ec="k", fc="C0", alpha=0.7)
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

# ╔═╡ 6236b296-5f60-11eb-3aa7-5188433b3906
md"# matrix viz"

# ╔═╡ 5ae47630-5f64-11eb-39f8-654f8d277674
function viz_matrix(A::Array{Union{Float64, Missing}, 2})
	norm = PyPlot.matplotlib.colors.Normalize(vmin=-4.0, vmax=4.0)
	cmap = PyPlot.matplotlib.cm.get_cmap("PiYG") # diverging
	
	# mapping adsorption properties to colors
	function a_to_color(a::Union{Float64, Missing})
		if ismissing(a)
			return (0.5, 0.5, 0.5, 1.0)
		else
			return cmap(norm(a))
		end
	end
	
	figure(figsize=(4, 15))
	ax = gca()
	is = imshow(a_to_color.(A))
	xlabel("gas\nadsorption\nproperties")
	ylabel("COFs")
	xticks([])
	yticks([])
	ylim([-0.5, size(A)[1]-0.5])
	xlim([-0.5, size(A)[2]-0.5])
	colorbar(PyPlot.matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
		label=L"$A_{ij}$ (normalized)", extend="both", shrink=0.4)
	tight_layout()
	savefig("matrix_example.pdf", format="pdf")
	gcf()
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
    glrm = GLRM(At, QuadLoss(), rp, rm, hp.k+1, obs=obs, offset=true)
#    init_svd!(glrm)
    P, M, ch = fit!(glrm)
    @assert size(P) == (hp.k + 1, n_p)
    @assert size(M) == (hp.k + 1, n_m)

    @assert isapprox(impute(glrm), P' * M)
    return P, M, glrm, ch
end

# ╔═╡ 21995e36-5f69-11eb-0a95-13d0136099df
function fit_bias_only_glrm(At::Array{Union{Float64, Missing}, 2},
		                     obs::Array{Tuple{Int64,Int64}, 1})
	hp = HyperParam(0, 0.0)
	return fit_glrm(At, hp, obs)
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
	ks = collect(1:5)                       # ranks
	λs = 10.0 .^ range(-1.0, 3.0, length=5) # regularization params
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
		G, P, glrm, ch = fit_glrm(At, hyper_param, ids_train)
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
struct Result
	# the data
	A::Array{Union{Float64, Missing}, 2}
	# the depolyment model model
	M::Array{Float64, 2}
	P::Array{Float64, 2}
	hp::HyperParam
	# true test data pts
	a::Array{Float64, 1}
	# pred test data pts
	â::Array{Float64, 1}
	# pred test data for bias only
	âb::Array{Float64, 1}
	# gas-wise Spearman rank coeffs
	ρp::Array{Float64, 1}
	# gas-wise Spearman rank coeffs bias only
	ρpb::Array{Float64, 1}
end

# ╔═╡ b285e2fe-5ba7-11eb-2e12-83e72bcafa2f
# 1. simulate data collection
# 2. split observed values into train and validation set
# 3. remaining unobserved = test data
# 4. do hyperparam sweep to find optimal hyperparams, judged by validation data
# 5. now that we hv optimal hyper params, retrain on all observed.
# 6. evaluate it on test entries
#   return (true values, pred values)
function run_simulation(θ::Float64)
	###
	#    set up data
	###
	A, θ_true = sim_data_collection(θ)

	# store for later.
	μs, σs = normalize!(A)

	At = collect(A')
	
	# split into test/validation/train
	ids_obs, ids_unobs = ids_obs_and_unobs(At)
	ids_test = ids_unobs
	@assert(all(ismissing.([At[p, m] for (p, m) in ids_test])))
	ids_train, ids_valid = train_test_split(ids_obs, test_size=0.2, shuffle=true)

	###
	#   hyper-parameter sweep using train/valid data 
	#        to find optimal hyper params (k, λ)
	###
	opt_valid_run = hyperparam_sweep(hyper_params, At, ids_train, ids_valid)

	###
	#   deployment time: train model on all observed data
	###
	P, M, glrm, ch = fit_glrm(At, opt_valid_run.hyper_param, ids_obs)
	Gb, Mb, glrmb, chb = fit_bias_only_glrm(At, ids_obs)
	
	###
	#   test time: test depolyed model on unobserved entries
	###
	Ât = impute(glrm)
	Âtb = impute(glrmb)

	# compute truth from normalizations
	At_complete = compute_At_complete(μs, σs)

	# test model on unobserved
	a = [At_complete[p, m] for (p, m) in ids_test]
	â = [Ât[p, m] for (p, m) in ids_test]
	âb = [Âtb[p, m] for (p, m) in ids_test]
	# get spearman ranks for gases

	ρp = [ρ_p(p, At_complete, Ât, ids_test) for p = 1:n_p]
	ρpb = [ρ_p(p, At_complete, Âtb, ids_test) for p = 1:n_p]
	return Result(A, M, P, opt_valid_run.hyper_param,
		          a, â, âb, ρp, ρpb)
end

# ╔═╡ 5d38b414-5f41-11eb-14b9-73a9007fc263
md"# $\theta=0.4$ example"

# ╔═╡ 4f81a520-5f6d-11eb-1960-9918ca4f25e9
res = run_simulation(0.4)

# ╔═╡ 68fe93ae-5f6e-11eb-012a-81378cd15b41
viz_matrix(res.A)

# ╔═╡ 53585188-5f6f-11eb-0fc0-abbd20ee33fe
begin
	# clip values
	da_cutoff = 6.0
	δ = 0.1
	for 🐶 in [res.a, res.â]
		🐶[🐶 .< - da_cutoff] .= - da_cutoff
		🐶[🐶 .>   da_cutoff] .=   da_cutoff
	end
	
	# parity plot
	figure()
	hexbin(res.a, res.â, mincnt=1, bins="log")
	xlabel(L"true $A_{ij}$ (normalized)")
	ylabel(L"predicted $A_{ij}$ (normalized)")
	xlim([-da_cutoff - δ, da_cutoff + δ])
	ylim([-da_cutoff - δ, da_cutoff + δ])
	plot([-da_cutoff, da_cutoff], [-da_cutoff, da_cutoff], 
		linestyle="--", color="gray")
	text(4, -4, 
		@sprintf("hyperparameters:\nk = %d\nλ = %.2f\n\nperformance:\nρ = %.2f\nRMSE = %.2f", res.hp.k, res.hp.λ, corspearman(res.a, res.â), rmsd(res.a, res.â)),
		ha="center", va="center"
		)
	colorbar(label="# (COF, adsorption property) pairs")
	gca().set_aspect("equal", "box")
	tight_layout()
	savefig("parity_plot.pdf", format="pdf")
	gcf()
end

# ╔═╡ 74068408-5f70-11eb-02ba-417e847034c4
begin
	figure(figsize=(10, 4.8))
	bar(1:n_p, res.ρp, label=@sprintf("k = %d, λ = %.2f", res.hp.k, res.hp.λ))
	scatter(1:n_p, res.ρpb, marker="*", zorder=100, s=150, 
		    ec="k", label="k = 0"
		    )
	xticks(1:n_p, [prop_to_label[p] for p in properties], rotation=90)
	ylabel("Spearman's rank\ncorrelation coefficient\n"  * L"$\rho$")
	# text(12.0, 0.9,
	# 	@sprintf("hyperparameters:\nk = %d\nλ = %.2f", res.hp.k, res.hp.λ),
	# 	ha="center", va="center"
	# 	)
	legend()
	ylim([0.0, 1.0])
	gcf()
end

# ╔═╡ 8548a48c-5f73-11eb-3d4f-550078ec546a
begin
	μ = res.M[end, :]
	ids_sort = sortperm(μ, rev=true)
	
	n_show = 20
	n_space = 3
	
	ids_best  = 1:n_show
	ids_worst = (n_show + n_space + 1):(2 * n_show + n_space + 1)
	
	figure(figsize=(10.0, 4.8))
	bar(ids_best , μ[ids_sort][1:n_show])
	bar(ids_worst, μ[ids_sort][end-n_show:end])
	xticks(vcat(ids_best, ids_worst), 
		   vcat(materials[ids_sort][ids_best], materials[ids_sort][ids_worst]),
		   rotation=90
		  )
	xlim([0, 2 * n_show + n_space + 2])
	# ylim([-4.25, 4.25])
	xlabel("COF")
	ylabel(L"material bias, $\mu_i$")
	
	scatter((n_show+1):(n_show+n_space), zeros(3), color="k")
	
	tight_layout()
	savefig("material_bias.pdf", format="pdf")
	gcf()
end

# ╔═╡ b0560c02-5f80-11eb-338b-c9cc48b741af
md"### learn latent space

of materials and properties together.
"

# ╔═╡ ba8ce81e-5f80-11eb-3e39-f942cb6d0d1f
X = hcat(res.M[1:end-1, :], res.P[1:end-1, :])

# ╔═╡ c6caaa48-5f7f-11eb-3853-fdffcd51b2d5
begin
	# input: (a column-major matrix of shape (n_features, n_samples))
	latent_space = umap(X, 2)
	m_vecs = latent_space[:, 1:n_m]
	p_vecs = latent_space[:, (n_m+1):end]
	@assert size(p_vecs) == (2, n_p)
end

# ╔═╡ 59a72a22-5f82-11eb-1424-0913e7830bc4
begin
	figure()
	scatter(m_vecs[1, :], m_vecs[2, :])
	xlabel("UMAP dimension 1")
	ylabel("UMAP dimension 2")
	gca().set_aspect("equal", "box")
	tight_layout()
	gcf()
end

# ╔═╡ 0cd6cd76-5f6e-11eb-0bf5-2f0ea61ef29b
md"# loop over θs"

# ╔═╡ 5bbe8438-5f41-11eb-3d16-716bcb25400b
begin
	function run_θ_study(θ::Float64, nb_sims::Int)
		# run sims, keep track of true vs pred and (k, λ)
		a = Float64[] # true adsorption
		â = Float64[] # pred adsorption
		k = Int[]     # rank
		λ = Float64[] # reg param
		for s = 1:nb_sims
			# run sim
			test_result, opt_valid_run = run_simulation(θ)
			a = vcat(a, test_result.a)
			â = vcat(â, test_result.â)
			push!(k, opt_valid_run.hyper_param.k)
			push!(λ, opt_valid_run.hyper_param.λ)
		end
		return a, â, k, λ
	end

	a, â, k, λ = run_θ_study(0.4, 1)
end

# ╔═╡ 15c50a00-5f42-11eb-2044-91cd5ccf6d67
md"
stack vectors together to get the true distribution over different #'s of simulations.
"

# ╔═╡ b329a5f4-5f46-11eb-2e06-e9e991991815
begin
	# dist'n of hyper-params
	fig, axs = subplots(2, 1)
	for kᵢ in ks
		axs[1].plot([kᵢ, kᵢ], [0, sum(k .== kᵢ)], color="C0", lw=3)
		axs[1].scatter(kᵢ, sum(k .== kᵢ), color="C0")
	end
	axs[1].set_xlabel(L"dimensionality of latent space, $k$")
	axs[1].set_ylabel("# simulations")
	axs[1].set_xticks(ks)
	
	for λᵢ in λs
		axs[2].plot([λᵢ, λᵢ], [0, sum(λ .== λᵢ)], color="C1", lw=3)
		axs[2].scatter(λᵢ, sum(λ .== λᵢ), color="C1")
	end
	axs[2].set_xlabel(L"regularization parameter, $\lambda$")
	axs[2].set_ylabel("# simulations")
	axs[2].set_xticklabels(ks)
	axs[2].set_xscale("log")
	
	# ax.set_yscale("log")
	# ax.set_ylabel(L"$\lambda$")
	# ax.set_yticks(1:length(λs))
	tight_layout()
	suptitle("hyper-parameter distribution")
	gcf()
end

# ╔═╡ f32146b2-5f5f-11eb-019a-0b5838c2fd40


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
# ╟─6236b296-5f60-11eb-3aa7-5188433b3906
# ╠═5ae47630-5f64-11eb-39f8-654f8d277674
# ╟─6ef474cc-5b90-11eb-2fe2-87bc7b48f5b7
# ╠═8a3c55ae-5ba4-11eb-3354-f9a8feaa7e91
# ╠═1d745ea2-5ba5-11eb-274a-4ff7a65b0cb6
# ╠═cc771288-5ba6-11eb-1792-9dfc54c58a8c
# ╠═f7158eea-5ba4-11eb-17cc-d57c477cec39
# ╠═8152d710-5b90-11eb-39f5-45d81aa298ab
# ╠═21995e36-5f69-11eb-0a95-13d0136099df
# ╟─e2dd554c-5baf-11eb-1b49-654d19bedecc
# ╠═168d4c44-5bb3-11eb-2d16-af69f613b625
# ╠═36da7c1e-5bb1-11eb-2bc9-438dabcf9cc5
# ╟─efc74f24-5ba0-11eb-2c44-6dac87ec534a
# ╠═a269ab26-5ba4-11eb-1001-6703f57f495c
# ╠═9cf75472-5ba0-11eb-371a-5bc338946b61
# ╟─2009751a-5bae-11eb-158f-a3d9cb98fe24
# ╠═09b74408-5baa-11eb-3735-e9333756f565
# ╠═b285e2fe-5ba7-11eb-2e12-83e72bcafa2f
# ╟─5d38b414-5f41-11eb-14b9-73a9007fc263
# ╠═4f81a520-5f6d-11eb-1960-9918ca4f25e9
# ╠═68fe93ae-5f6e-11eb-012a-81378cd15b41
# ╠═53585188-5f6f-11eb-0fc0-abbd20ee33fe
# ╠═74068408-5f70-11eb-02ba-417e847034c4
# ╠═8548a48c-5f73-11eb-3d4f-550078ec546a
# ╟─b0560c02-5f80-11eb-338b-c9cc48b741af
# ╠═ba8ce81e-5f80-11eb-3e39-f942cb6d0d1f
# ╠═c6caaa48-5f7f-11eb-3853-fdffcd51b2d5
# ╠═59a72a22-5f82-11eb-1424-0913e7830bc4
# ╟─0cd6cd76-5f6e-11eb-0bf5-2f0ea61ef29b
# ╠═5bbe8438-5f41-11eb-3d16-716bcb25400b
# ╟─15c50a00-5f42-11eb-2044-91cd5ccf6d67
# ╠═b329a5f4-5f46-11eb-2e06-e9e991991815
# ╠═f32146b2-5f5f-11eb-019a-0b5838c2fd40
