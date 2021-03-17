### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 92502b2a-7f83-11eb-152b-f10d7015d5cc
begin
	using Pkg; Pkg.activate("Project.toml"); Pkg.instantiate()
	
	using LowRankModels, CSV, DataFrames, PyPlot, Statistics, Distributions, StatsBase, Printf, UMAP, PyCall, ProgressMeter, Random
	using ScikitLearn.CrossValidation: train_test_split
	
	PyPlot.matplotlib.style.use("https://gist.githubusercontent.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea/raw/64a78942d3f7b4b5054902f2cee84213eaff872f/matplotlibrc")
	
	adjustText = pyimport("adjustText")
	sbn = pyimport("seaborn")
	
	const my_seed = 97330
	Random.seed!(my_seed)
end

# ╔═╡ ae415fa4-5b82-11eb-0051-072097bb0439
md"# read in data"

# ╔═╡ b3deec9e-5b82-11eb-0e37-abd2ac9d4b44
begin
	df = CSV.read("aiida_ads_data_oct20.csv", DataFrame)
	df[:, :cof] = map(c -> split(c, ".cif")[1], df[:, :cof])
	
	# log-10 transform the henry coefficients.
	for henry_col in Symbol.(["h2o_henry", "h2s_henry", "xe_henry", "kr_henry"])
	    # convert units from mol/(kg-Pa) to mol/(kg-bar)
	    df[!, henry_col] = 1e5 * df[:, henry_col]
	    # log-10 transform
	    df[!, henry_col] = log10.(df[:, henry_col])
	end
	
	# cof common names
	#  https://github.com/danieleongari/CURATED-COFs/blob/master/cof-frameworks.csv
	df_names = CSV.read("cof-frameworks.csv", DataFrame)
	rename!(df_names, Symbol("CURATED-COFs ID") => :cof)
	df_names = df_names[:, [:cof, :Name]]
	
	df = leftjoin(df, df_names, on=:cof)
	
	first(df, 10)
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
	                     "h2s_henry"      => L"H$_2$S Henry" * "\n(300 K)", 
	                     "xe_henry"       => "Xe Henry\n (300 K)", 
	                     "kr_henry"       => "Kr Henry\n (300 K)", 
	                     "ch4_65bar"      => L"CH$_4$" * "\n(298 K, 65 bar)", 
	                     "ch4_5.8bar"     => L"CH$_4$" * "\n(298 K, 5.8 bar)"
	                    )
	
	prop_to_label2 = Dict("o2_5bar"        => L"O$_2$" * "\n298 K\n5 bar", 
						 "o2_140bar"      => L"O$_2$" * "\n298 K\n140 bar", 
						 "co2_0.001bar"   => L"CO$_2$" * "\n300 K\n0.001 bar", 
						 "co2_30bar"      => L"CO$_2$" * "\n300 K\n30 bar", 
						 "n2_0.001bar"    => L"N$_2$" * "\n300 K\n0.001 bar", 
						 "n2_30bar"       => L"N$_2$" * "\n300 K\n30 bar", 
						 "h2_77K_5bar"    => L"H$_2$" * "\n77 K\n5 bar", 
						 "h2_77K_100bar"  => L"H$_2$" * "\n77 K\n100 bar", 
						 "h2_298K_5bar"   => L"H$_2$" * "\n298 K\n5 bar", 
						 "h2_298K_100bar" => L"H$_2$" * "\n298 K\n100 bar", 
						 "h2o_henry"      => L"H$_2$O Henry" * "\n300 K", 
						 "h2s_henry"      => L"H$_2$S Henry" * "\n300 K", 
						 "xe_henry"       => "Xe Henry\n300 K", 
						 "kr_henry"       => "Kr Henry\n300 K", 
						 "ch4_65bar"      => L"CH$_4$" * "\n298 K\n65 bar", 
						 "ch4_5.8bar"     => L"CH$_4$" * "\n298 K\n5.8 bar"
						)

	const materials = df[:, :Name]# map(x -> split(x, ".cif")[1], df[:, :cof])
	const n_m = length(materials)
	const n_p = length(properties)
	# material property matrix
	const A_complete_unnormalized = convert(Matrix, df[:, properties])
	
	# normalize columns
	A_n = deepcopy(A_complete_unnormalized)
	for i = 1:n_p
		A_n[:, i] = (A_n[:, i] .- mean(A_n[:, i])) ./ std(A_n[:, i])
	end
	# dataframe form
	df_n = DataFrame()
	for i = 1:n_p
		df_n[:, prop_to_label2[properties[i]]] = A_n[:, i]
	end
	CSV.write("normalized_props.csv", df_n)
	df_n
end

# ╔═╡ 6713990c-5b8d-11eb-2196-7182f36cad59
md"# simulate data collection"

# ╔═╡ 2931005c-5b8d-11eb-2375-5dacf441be72
# θ: target fraction observed values
function sim_data_collection(θ::Float64)
	nb_observed = floor(Int, θ * n_m * n_p)
	# sample observed tuples
    Ω = sample([(m, p) for m = 1:n_m, p = 1:n_p][:], nb_observed, replace=false)
	# construct incomplete matrix
    A = zeros(Union{Float64, Missing}, n_m, n_p)
	fill!(A, missing) # default missing
	# fill in observed values
	for (m, p) in Ω
		A[m, p] = A_complete_unnormalized[m, p]
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
function viz_matrix(A::Array{Union{Float64, Missing}, 2}; θ::Float64)
	norm = PyPlot.matplotlib.colors.Normalize(vmin=-3.0, vmax=3.0)
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
	is = imshow(a_to_color.(A), interpolation="None")
	xlabel("gas\nadsorption\nproperties")
	ylabel("COFs")
	xticks([])
	yticks([])
	ylim([-0.5, size(A)[1]-0.5])
	xlim([-0.5, size(A)[2]-0.5])
	colorbar(PyPlot.matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
		label=L"$A_{mp}$ (standardized)", extend="both", shrink=0.4)
	title(@sprintf("θ = %.1f", θ), fontsize=12)
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

# ╔═╡ b518d378-65ca-11eb-3bda-adcd26ccaa13
begin
	struct HPGrid
		ks::Array{Int, 1}
		λs::Array{Float64, 1}
		n::Int
	end
	
	HPGrid(ks, λs) = HPGrid(ks, sort(λs), length(ks) * length(λs))
end

# ╔═╡ 1d745ea2-5ba5-11eb-274a-4ff7a65b0cb6
struct ValidationRun
	hyper_param::HyperParam
	rmsd::Float64
	ρ::Float64
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

# ╔═╡ 080ce23a-6e32-11eb-14b0-c3d43725f78a
function max_ρ(vrs::Array{ValidationRun, 1})
	da_best = vrs[1]
	for i = 2:length(vrs)
		if vrs[i].ρ > da_best.ρ
			da_best = vrs[i]
		end
	end
	return da_best
end

# ╔═╡ 8152d710-5b90-11eb-39f5-45d81aa298ab
# k = rank of matrix
# λ = regularization param
# obs = which observations we train on.
# (P₀, M₀) ==  initial guess for speed up.
# gotta pass the transpose for the bias to work.
function fit_glrm(At::Array{Union{Float64, Missing}, 2}, 
		          hp::HyperParam,
		          obs::Array{Tuple{Int64,Int64}, 1};
				  P₀::Union{Array{Float64, 2}, Nothing}=nothing, 
		          M₀::Union{Array{Float64, 2}, Nothing}=nothing)
	# quadratic regularizors
    rp = QuadReg(hp.λ / n_p)
    rm = QuadReg(hp.λ / n_m)
	# this should be the transpose...
	@assert size(At) == (n_p, n_m)
	
	# guess, same code as glrm.jl
	if isnothing(P₀) P₀ = randn(hp.k + 1, n_p) end# initialize
	if isnothing(M₀) M₀ = randn(hp.k + 1, n_m) end
	
	# lrm.jl: X'*Y, where X is a kxm matrix and Y is a kxn matrix
	# A = M' P
	# At = P' M
	#   so here X is the analogy of P; Y is the analogy of M
    glrm = GLRM(At, QuadLoss(), rp, rm, hp.k+1, 
		        obs=obs, offset=true, X=P₀, Y=M₀)
#    init_svd!(glrm)
    P, M, ch = fit!(glrm)
    @assert size(P) == (hp.k + 1, n_p)
    @assert size(M) == (hp.k + 1, n_m)

    @assert isapprox(impute(glrm), P' * M)
    return P, M, glrm, ch
end

# ╔═╡ 3aba3150-65cd-11eb-2c51-878ef71193ac
# begin
# 	θ = 0.3
# 	A, θ_true = sim_data_collection(θ)
	
# 	# store for later.
# 	μs, σs = normalize!(A)
	
# 	At = collect(A')
# 	hp = HyperParam(3, 0.4)
	
# 	P₀ = randn(hp.k, n_p) # initialize
# 	M₀ = randn(hp.k, n_m)
	
# 	fit_glrm(At, hp, observations(At), P₀=P₀, M₀=M₀)
# end

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
hpgrid = HPGrid(collect(1:15),                       # ranks
				10.0 .^ range(1.0, 3.0, length=25)  # reg params
				)

# ╔═╡ 2ff3b2f6-65db-11eb-156d-7fbcc6c79e76


# ╔═╡ 9cf75472-5ba0-11eb-371a-5bc338946b61
# hyperparam sweep
# return optimal hyperparams with min rmsd
function hyperparam_sweep(hpgrid::HPGrid, 
						  At::Array{Union{Float64, Missing}, 2},
						  ids_train::Array{Tuple{Int64,Int64},1},
						  ids_valid::Array{Tuple{Int64,Int64},1};
						  show_progress::Bool=false
						 )
	pm = Progress(hpgrid.n)
	valid_runs = ValidationRun[]
	sweep_no = 0
	# order in loop is important
	for k in hpgrid.ks
		# for regularization path, start with smallest lambda
		#     ;don't want to zero out any latent reps!
		P = randn(k + 1, n_p) # initialize
		M = randn(k + 1, n_m)
		for λ in hpgrid.λs
			sweep_no += 1
			hyper_param = HyperParam(k, λ)
			
			if show_progress
				update!(pm, sweep_no)
			end
			# train on training data
			P, M, glrm, ch = fit_glrm(At, hyper_param, ids_train, P₀=P, M₀=M)
			# impute missing entries
			Ât = impute(glrm)
			# evaluate on validation data
			a = [At[p, m] for (p, m) in ids_valid]
			â = [Ât[p, m] for (p, m) in ids_valid]

			push!(valid_runs, ValidationRun(hyper_param, 
											rmsd(a, â),
											corspearman(a, â)
										   )
				  )
		end
	end
	# return min_rmse(valid_runs)
	return max_ρ(valid_runs)
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
function run_simulation(θ::Float64; show_progress::Bool=false)
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
	ids_train, ids_valid = train_test_split(ids_obs, test_size=0.2, 
		shuffle=true, random_state=floor(Int, my_seed + 100 * θ))

	###
	#   hyper-parameter sweep using train/valid data 
	#        to find optimal hyper params (k, λ)
	###
	opt_valid_run = hyperparam_sweep(hpgrid, At, ids_train, ids_valid, show_progress=show_progress)

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

# ╔═╡ 128a9fa0-808b-11eb-22f0-afd1dd30593c
θ = 0.4

# ╔═╡ bdb51f2e-65d8-11eb-1063-614c91a95e6e


# ╔═╡ 4f81a520-5f6d-11eb-1960-9918ca4f25e9
@time res = run_simulation(0.4, show_progress=true)

# ╔═╡ 68fe93ae-5f6e-11eb-012a-81378cd15b41
viz_matrix(res.A, θ=θ)

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
	xlabel(L"true $A_{mp}$ (standardized)")
	ylabel(L"predicted $A_{mp}$ (standardized)")
	xlim([-da_cutoff - δ, da_cutoff + δ])
	ylim([-da_cutoff - δ, da_cutoff + δ])
	plot([-da_cutoff, da_cutoff], [-da_cutoff, da_cutoff], 
		linestyle="--", color="gray")
	# text(4, -4, 
	legend(title=@sprintf("θ = %.1f\n\nhyperparams:\nk = %d\nλ = %.2f\n\nperformance:\nρ = %.2f\nRMSE = %.2f", θ, res.hp.k, res.hp.λ, corspearman(res.a, res.â), rmsd(res.a, res.â)))
		# ha="center", va="center"
		# )
	colorbar(label="# (COF, adsorption property) pairs")
	gca().set_aspect("equal", "box")
	tight_layout()
	savefig("parity_plot.pdf", format="pdf")
	gcf()
end

# ╔═╡ 74068408-5f70-11eb-02ba-417e847034c4
begin
	# plot in sorted order
	ids_props_sorted = sortperm(res.ρp, rev=true)
		
	figure(figsize=(10, 4.8))
	bar(1:n_p, res.ρp[ids_props_sorted], 
		label=@sprintf("θ = %.1f, k = %d, λ = %.2f", θ, res.hp.k, res.hp.λ)
	)
	scatter(1:n_p, res.ρpb[ids_props_sorted], marker="*", zorder=100, s=160, 
		    ec="k", label=@sprintf("θ = %.1f, k = 0", θ)
	)
	xlim([0.5, n_p+0.5])
	xticks(1:n_p, [prop_to_label[p] for p in properties[ids_props_sorted]], rotation=90)
	ylabel("Spearman's rank\ncorrelation coefficient\n"  * L"$\rho$")
	# text(12.0, 0.9,
	# 	@sprintf("hyperparameters:\nk = %d\nλ = %.2f", res.hp.k, res.hp.λ),
	# 	ha="center", va="center"
	# 	)
	legend()
	ylim([-0.1, 1.0])
	tight_layout()
	savefig("rho_per_gas.pdf", format="pdf")
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
	bar(ids_best , μ[ids_sort][1:n_show], color="C3")
	bar(ids_worst, μ[ids_sort][end-n_show:end], color="C4")
	xticks(vcat(ids_best, ids_worst), 
		   vcat(materials[ids_sort][1:n_show], materials[ids_sort][n_m-n_show:n_m]),
		   rotation=90
		  )
	xlim([0, 2 * n_show + n_space + 2])
	# ylim([-4.25, 4.25])
	xlabel("COF")
	ylabel(L"material bias, $\mu_i$")
	legend(title=@sprintf("θ = %.1f\n\nhyperparams:\nk = %d\nλ = %.2f", θ, res.hp.k, res.hp.λ))
	
	scatter((n_show+1):(n_show+n_space), zeros(3), color="k")
	
	tight_layout()
	savefig("material_bias.pdf", format="pdf")
	gcf()
end

# ╔═╡ ce346a40-667c-11eb-03d3-eb7c4510ff26
df[ids_sort, [:cof, :Name]][1:3, :]

# ╔═╡ 1568fe16-667e-11eb-0ecc-bfd712234906
df[ids_sort, [:cof, :Name]][end-2:end, :]

# ╔═╡ b0560c02-5f80-11eb-338b-c9cc48b741af
md"### learn latent space

of materials and properties together.
"

# ╔═╡ ba8ce81e-5f80-11eb-3e39-f942cb6d0d1f
X = hcat(res.M[1:end-1, :], res.P[1:end-1, :])

# ╔═╡ c6caaa48-5f7f-11eb-3853-fdffcd51b2d5
begin
	m_vecs = res.M[1:end-1, :]
	p_vecs = res.P[1:end-1, :]
	if res.hp.k > 2
		# input: (a column-major matrix of shape (n_features, n_samples))
		latent_space = umap(X, 2)
		m_vecs = latent_space[:, 1:n_m]
		p_vecs = latent_space[:, (n_m+1):end]
		@assert size(p_vecs) == (2, n_p)
	end
end

# ╔═╡ 8024beae-5f88-11eb-3e97-b7afbbbc6f5c
function viz_prop_latent_space()
	cs = sbn.color_palette("husl", 16)
	
	figure(figsize=(8, 8))
	
	if res.hp.k > 2
		xlabel("UMAP dimension 1")
		ylabel("UMAP dimension 2")
	else
		xlabel("latent dimension 1")
		ylabel("latent dimension 2")
	end
	texts = []
	for p = 1:n_p
		scatter(p_vecs[1, p], p_vecs[2, p], edgecolor="k", color=cs[p])
		push!(texts, 
			annotate(prop_to_label[properties[p]], 
				(p_vecs[1, p], p_vecs[2, p]), 
				fontsize=10, ha="center", color=cs[p]
				# arrowprops=Dict(:facecolor="gray", :shrink=0.05)
			)
			)
	end
	adjustText.adjust_text(texts, force_text=0.01, force_points=0.01)
	# text(-2, 4.5, 
	# 	@sprintf("hyperparameters:\nk = %d\nλ = %.2f", res.hp.k, res.hp.λ),
	# 	ha="center", va="center", fontsize=20)
	legend(title=@sprintf("θ = %.1f\n\nk = %d\nλ = %.2f", θ, res.hp.k, res.hp.λ))
	axvline(x=0.0, color="lightgray", zorder=0)
	axhline(y=0.0, color="lightgray", zorder=0)
	# colorbar(label=prop_to_label[properties[p]], extend="both")
	gca().set_aspect("equal", "box")
	tight_layout()
	savefig("prop_latent_space.pdf", format="pdf")
	gcf()
end

# ╔═╡ ab3a5568-5f88-11eb-373a-2f79bfce3cff
viz_prop_latent_space()

# ╔═╡ 59a72a22-5f82-11eb-1424-0913e7830bc4
function color_latent_material_space(p::Int)
	figure()
	scatter(m_vecs[1, :], m_vecs[2, :], c=A_n[:, p], s=25, 
		vmin=-3.0, vmax=3.0, cmap="PiYG", edgecolor="k")
	if res.hp.k > 2
		xlabel("UMAP dimension 1")
		ylabel("UMAP dimension 2")
	else
		xlabel("latent dimension 1")
		ylabel("latent dimension 2")
	end
	axvline(x=0.0, color="lightgray", zorder=0)
	axhline(y=0.0, color="lightgray", zorder=0)
	colorbar(label=prop_to_label[properties[p]], extend="both")
	# text(-3, 4.5, 
	# 	@sprintf("hyperparameters:\nk = %d\nλ = %.2f", res.hp.k, res.hp.λ),
	# 	ha="center", va="center")
	legend(title=@sprintf("θ = %.1f\nk = %d\nλ = %.2f", θ, res.hp.k, res.hp.λ))
	gca().set_aspect("equal", "box")

	savefig("latent_mat_space_$p.pdf", format="pdf")
	tight_layout()
	gcf()
end

# ╔═╡ b0619008-5f86-11eb-11b6-c7a3c4db9fd3
color_latent_material_space(15)

# ╔═╡ 86b00b60-5f89-11eb-071f-bb364af09c2a
color_latent_material_space(12)

# ╔═╡ ccf77948-86b5-11eb-28ce-69ff4f7c2f20
color_latent_material_space(11)

# ╔═╡ 244ce106-65e4-11eb-080b-f52f27e435fc
function color_latent_material_space_all()
	fig, axs = subplots(4, 4, sharex=true, sharey=true, figsize=(13, 11))
	p = 0
	for i = 1:4
		for j = 1:4
			p += 1
			da_plot = axs[i, j].scatter(m_vecs[1, :], m_vecs[2, :], c=A_n[:, p], s=25, 
							  vmin=-3.0, vmax=3.0, cmap="PiYG", edgecolor="k")
			# xlabel("UMAP dimension 1")
			# ylabel("UMAP dimension 2")
			axs[i, j].set_title(prop_to_label[properties[p]])
			axs[i, j].set_aspect("equal", "box")
			if (i == 4) & (j == 4)
				cb_ax = fig.add_axes([1.025, 0.1, 0.02, 0.8])
				fig.colorbar(da_plot, cax=cb_ax, extend="both", 
					label="standardized value")
				
				# fig.colorbar(da_plot, ax=axs[:, :], shrink=0.6)
			end
			axs[i, j].axvline(x=0.0, color="lightgray", zorder=0)
			axs[i, j].axhline(y=0.0, color="lightgray", zorder=0)
		end
	end
	suptitle("COF map colored by adsorption properties")
	tight_layout()
	savefig("latent_mat_space.pdf", format="pdf", bbox_inches="tight")
	gcf()
end

# ╔═╡ 2e523504-65e4-11eb-1cbc-fd2cb39afed6
color_latent_material_space_all()

# ╔═╡ 55ee1330-6508-11eb-37d1-1973f7e077ed
md"todo: color by void fraction etc."

# ╔═╡ 0cd6cd76-5f6e-11eb-0bf5-2f0ea61ef29b
md"# loop over θs"

# ╔═╡ 5bbe8438-5f41-11eb-3d16-716bcb25400b
begin
	struct SparseResult
		ρ::Float64
		ρb::Float64
		ρp::Array{Float64, 1}
		ρpb::Array{Float64, 1}
		k::Int
		λ::Float64
	end
	
	function run_θ_study(θ::Float64, nb_sims::Int)
		results = SparseResult[]
		for s = 1:nb_sims
			res = run_simulation(θ)
			push!(results, 
				SparseResult(corspearman(res.a, res.â),
				             corspearman(res.a, res.âb),
							 res.ρp,
							 res.ρpb,
					         res.hp.k,
							 res.hp.λ
						     )
				)
		end
		return results
	end
	
	nb_sims = 50
	θs = 0.1:0.1:0.9
	θresults = []
	pm = Progress(length(θs))
	for (i, θ) in enumerate(θs)
		push!(θresults, run_θ_study(θ, nb_sims))
		update!(pm, i)
	end
end

# ╔═╡ c7aa89b0-5f93-11eb-0503-5565bba9cb86
function viz_ρp_vsθ()
	fig, axs = plt.subplots(nrows=2, ncols=8, figsize=(13, 5.5), sharex=true)
	for (p, ax) in enumerate(axs)
		if p == 1 || p == 2
			ax.set_ylabel("Spearman's rank\ncorrelation coefficient\n" * L"$\rho$")
		else
			ax.spines["left"].set_visible(false)
			# ax.yaxis.set_visible(false)
			ax.set_yticklabels([])
		end
		if p % 2 == 0
			ax.set_xlabel(L"$\theta$")
		end
		ax.spines["right"].set_visible(false)
		ax.spines["bottom"].set_position("zero")
		ax.spines["top"].set_visible(false)
		ax.set_ylim([0.0, 1.0])
		
		# one per θ
		ρ_avg = [mean([res.ρp[p] for res in θresults[i]]) for i = 1:length(θs)]
		ρ_std = [std([res.ρp[p] for res in θresults[i]]) for i = 1:length(θs)]
		ρb_avg = [mean([res.ρpb[p] for res in θresults[i]]) for i = 1:length(θs)]
		ρb_std = [std([res.ρpb[p] for res in θresults[i]]) for i = 1:length(θs)]
		
		ax.plot(θs, ρ_avg, marker="o", clip_on=false, markeredgecolor="k")
		ax.fill_between(θs, ρ_avg .- ρ_std, ρ_avg .+ ρ_std, alpha=0.3, clip_on=false)
		
		ax.plot(θs, ρb_avg, marker="*", clip_on=false, markeredgecolor="k")
		ax.fill_between(θs, ρb_avg .- ρb_std, ρb_avg .+ ρb_std, alpha=0.3)
		
		ax.set_title(prop_to_label2[properties[p]], fontsize=14)
		ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
	end
	for ax in axs
		ax.set_xticks([0.0, 0.5, 1.0])
	end
	
	tight_layout()
	savefig("rho_vs_theta.pdf", format="pdf")
    
	gcf()
end

# ╔═╡ 56bb9b5c-5f95-11eb-0d3f-97cd4b7a48a0
viz_ρp_vsθ()

# ╔═╡ 0830c1de-5f9e-11eb-132a-77b3084102b2
begin
	dfθ = DataFrame(θ=Float64[], k=Int[], λ=Float64[])
	for i = 1:length(θs)
		for r in θresults[i]
			push!(dfθ, [θs[i], r.k, r.λ])
		end
	end
	CSV.write("theta_k_lambda_table.csv", dfθ)
	dfθ
end

# ╔═╡ Cell order:
# ╠═92502b2a-7f83-11eb-152b-f10d7015d5cc
# ╟─ae415fa4-5b82-11eb-0051-072097bb0439
# ╠═b3deec9e-5b82-11eb-0e37-abd2ac9d4b44
# ╟─bf9ed538-5b82-11eb-1198-3d35a209c5c0
# ╠═c5d42b2e-5b82-11eb-0631-35efb6b0800c
# ╟─6713990c-5b8d-11eb-2196-7182f36cad59
# ╠═2931005c-5b8d-11eb-2375-5dacf441be72
# ╠═a21ac3b8-5ba1-11eb-2d70-bdf4b395f563
# ╟─1d363324-5b8f-11eb-2c78-6980a0d5f110
# ╠═2366b3cc-5b8f-11eb-07e4-61cbc97d5480
# ╟─6236b296-5f60-11eb-3aa7-5188433b3906
# ╠═5ae47630-5f64-11eb-39f8-654f8d277674
# ╟─6ef474cc-5b90-11eb-2fe2-87bc7b48f5b7
# ╠═8a3c55ae-5ba4-11eb-3354-f9a8feaa7e91
# ╠═b518d378-65ca-11eb-3bda-adcd26ccaa13
# ╠═1d745ea2-5ba5-11eb-274a-4ff7a65b0cb6
# ╠═cc771288-5ba6-11eb-1792-9dfc54c58a8c
# ╠═080ce23a-6e32-11eb-14b0-c3d43725f78a
# ╠═8152d710-5b90-11eb-39f5-45d81aa298ab
# ╠═3aba3150-65cd-11eb-2c51-878ef71193ac
# ╠═21995e36-5f69-11eb-0a95-13d0136099df
# ╟─e2dd554c-5baf-11eb-1b49-654d19bedecc
# ╠═168d4c44-5bb3-11eb-2d16-af69f613b625
# ╠═36da7c1e-5bb1-11eb-2bc9-438dabcf9cc5
# ╟─efc74f24-5ba0-11eb-2c44-6dac87ec534a
# ╠═a269ab26-5ba4-11eb-1001-6703f57f495c
# ╟─2ff3b2f6-65db-11eb-156d-7fbcc6c79e76
# ╠═9cf75472-5ba0-11eb-371a-5bc338946b61
# ╟─2009751a-5bae-11eb-158f-a3d9cb98fe24
# ╠═09b74408-5baa-11eb-3735-e9333756f565
# ╠═b285e2fe-5ba7-11eb-2e12-83e72bcafa2f
# ╟─5d38b414-5f41-11eb-14b9-73a9007fc263
# ╠═128a9fa0-808b-11eb-22f0-afd1dd30593c
# ╟─bdb51f2e-65d8-11eb-1063-614c91a95e6e
# ╠═4f81a520-5f6d-11eb-1960-9918ca4f25e9
# ╠═68fe93ae-5f6e-11eb-012a-81378cd15b41
# ╠═53585188-5f6f-11eb-0fc0-abbd20ee33fe
# ╠═74068408-5f70-11eb-02ba-417e847034c4
# ╠═8548a48c-5f73-11eb-3d4f-550078ec546a
# ╠═ce346a40-667c-11eb-03d3-eb7c4510ff26
# ╠═1568fe16-667e-11eb-0ecc-bfd712234906
# ╟─b0560c02-5f80-11eb-338b-c9cc48b741af
# ╠═ba8ce81e-5f80-11eb-3e39-f942cb6d0d1f
# ╠═c6caaa48-5f7f-11eb-3853-fdffcd51b2d5
# ╠═8024beae-5f88-11eb-3e97-b7afbbbc6f5c
# ╠═ab3a5568-5f88-11eb-373a-2f79bfce3cff
# ╠═59a72a22-5f82-11eb-1424-0913e7830bc4
# ╠═b0619008-5f86-11eb-11b6-c7a3c4db9fd3
# ╠═86b00b60-5f89-11eb-071f-bb364af09c2a
# ╠═ccf77948-86b5-11eb-28ce-69ff4f7c2f20
# ╠═244ce106-65e4-11eb-080b-f52f27e435fc
# ╠═2e523504-65e4-11eb-1cbc-fd2cb39afed6
# ╟─55ee1330-6508-11eb-37d1-1973f7e077ed
# ╟─0cd6cd76-5f6e-11eb-0bf5-2f0ea61ef29b
# ╠═5bbe8438-5f41-11eb-3d16-716bcb25400b
# ╠═c7aa89b0-5f93-11eb-0503-5565bba9cb86
# ╠═56bb9b5c-5f95-11eb-0d3f-97cd4b7a48a0
# ╠═0830c1de-5f9e-11eb-132a-77b3084102b2
