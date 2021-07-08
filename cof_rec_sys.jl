### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 92502b2a-7f83-11eb-152b-f10d7015d5cc
begin
	using Pkg; Pkg.activate("Project.toml"); Pkg.instantiate()
	
	using LowRankModels, CSV, DataFrames, PyPlot, Statistics, Distributions, StatsBase, Printf, UMAP, PyCall, ProgressMeter, Random, Test, PlutoUI
	using ScikitLearn.CrossValidation: train_test_split
	
	PyPlot.matplotlib.style.use("https://gist.githubusercontent.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea/raw/64a78942d3f7b4b5054902f2cee84213eaff872f/matplotlibrc")
	
	adjustText = pyimport("adjustText")
	sbn = pyimport("seaborn")
	axes_grid1 = pyimport("mpl_toolkits.axes_grid1")
	
	const my_seed = 97330
	Random.seed!(my_seed);
end

# ╔═╡ 43935790-86c3-11eb-3b21-cde10d72855b


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
	# list of properties we wish to include in the rec sys
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
	              "ch4_65bar",
	              "ch4_5.8bar",
			      "h2o_henry", 
	              "h2s_henry",
	              "xe_henry",
	              "kr_henry",
	              ]
	
	# mapping the properties to pretty strings for making plots
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

	const materials = df[:, :Name] # names of the COFs
	const n_m = length(materials)  # number of materials
	const n_p = length(properties) # number of properties
end

# ╔═╡ 5a74052f-dee7-4d7c-9611-c8d5c09cc778
md"""
!!! warning "warning"
in this notebook, we will work with the transpose of the material-property matrix in the paper. the reason is that `LowRankModels.jl` only allows offsets to be added to the columns [source](https://web.stanford.edu/~boyd/papers/pdf/glrm.pdf). so we will normalize the rows and add offsets to the columns.

entry $(p, m)$ of the matrix $X:=A^\intercal$ in this code is entry $(m, p)$ of $A$ in the paper, which is property $p$ of material $m$.
"""


# ╔═╡ cd34bdd0-c427-407c-a7cd-155773b3a3c0
begin	
	# property-material matrix (un-normalized)
	const X = collect(convert(Matrix, df[:, properties])')
	@assert size(X) == (n_p, n_m)
	
	function write_normalized_data_for_seaborn()
		# normalize rows to have:
		#   mean zero
		#   unit variance
		X_n = deepcopy(X)
		for i = 1:n_p
			X_n[i, :] = (X_n[i, :] .- mean(X_n[i, :])) ./ std(X_n[i, :])
		end

		# write normalized properties to a .csv file 
		#   this is for visualization in seaborn_stuff.ipynb
		df_n = DataFrame()
		for p = 1:n_p
			df_n[:, prop_to_label2[properties[p]]] = X_n[p, :]
		end
		@assert length(df_n[:, 1]) == n_m
		CSV.write("normalized_props.csv", df_n)
	end
	write_normalized_data_for_seaborn()
end

# ╔═╡ 6713990c-5b8d-11eb-2196-7182f36cad59
md"# simulate data collection"

# ╔═╡ 2931005c-5b8d-11eb-2375-5dacf441be72
"""
simulate the process of data collection, by randomly selecting (property, material) pairs in the matrix to uncover.

# arguments
- θ: target fraction observed values

# returns
- X_θ: un-normalized property-material matrix with missing entries
- ids_obs:     list of tuples corresponding to observed entries
- ids_unobs: list of tuples corresponding to unobserved (missing) entries
"""
function sim_data_collection(θ::Float64)
	# number of observed entries
	nb_observed = floor(Int, θ * n_m * n_p)
	
	# sample observed tuples
    ids_obs = sample([(p, m) for p=1:n_p, m=1:n_m][:], nb_observed, replace=false)
	# the rest are unobserved
	ids_unobs = [(p, m) for p=1:n_p, m=1:n_m if !((p, m) in ids_obs)][:]
	
	# construct the un-normalized, incomplete, material-property matrix
    X_θ = zeros(Union{Float64, Missing}, n_p, n_m)
	fill!(X_θ, missing) # default missing
	# fill in observed values
	for (p, m) in ids_obs
		X_θ[p, m] = X[p, m]
	end
	
    return X_θ, ids_obs, ids_unobs
end

# ╔═╡ 44be5768-d5d6-468a-ba5b-541a7f1213a9
with_terminal() do 
	@testset "data collection" begin
		X_θ, ids_obs, ids_unobs = sim_data_collection(0.9)
		@test sum(.! ismissing.(X_θ)) / (n_m*n_p) ≈ 0.9
		
		X_θ_new, _, _ = sim_data_collection(0.9)
		@test ismissing.(X_θ_new) !== ismissing.(X_θ) # so it changes
		
		@test all(.! ismissing.([X_θ[p, m] for (p, m) in ids_obs]))
		@test all(   ismissing.([X_θ[p, m] for (p, m) in ids_unobs]))
	end
end

# ╔═╡ 1d363324-5b8f-11eb-2c78-6980a0d5f110
md"# normalization of columns"

# ╔═╡ 2366b3cc-5b8f-11eb-07e4-61cbc97d5480
"""
given an incomplete property-material matrix, normalize the rows such that they have mean zero and unit variance. modifies the matrix X_θ passed to it.

# arguments
- X_θ: incomplete property-material matrix

# returns
- μs: means of properties
- σs: standard deviations of properties
(both used for normalization of the rows. return these to allow normalization of test data.)
"""
function normalize!(X_θ::Array{Union{Float64, Missing}, 2})
	# store means and standard deviations of the properties
	μs = zeros(n_p)
	σs = zeros(n_p)
	# loop through properties.
    for p = 1:n_p
		# get the observed columns = materials with this property
        ids_obs = .! ismissing.(X_θ[p, :])
		# compute mean and std of property (using observed values)
		μs[p] = mean(X_θ[p, ids_obs])
		σs[p] =  std(X_θ[p, ids_obs])
		# actually normalize the row.
        X_θ[p, :] .= (X_θ[p, :] .- μs[p]) ./ σs[p]
    end
	return μs, σs
end

# ╔═╡ b6a8b717-0f44-4eba-bff9-073079687dd9
with_terminal() do 
	@testset "column normalization" begin
		X_θ, ids_obs, ids_unobs = sim_data_collection(0.9)
		
		# test using row 4.
		ids_not_missing = .! ismissing.(X_θ[4, :])

		μ4 = mean(X_θ[4, ids_not_missing])
		σ4 = std( X_θ[4, ids_not_missing])

		μs, σs = normalize!(X_θ)

		@test isapprox(mean(X_θ[4, ids_not_missing]), 0.0, atol=1e-10)
		@test isapprox( std(X_θ[4, ids_not_missing]), 1.0, atol=1e-10)
		@test isapprox(μs[4], μ4)
		@test isapprox(σs[4], σ4)
	end
end

# ╔═╡ 6236b296-5f60-11eb-3aa7-5188433b3906
md"# matrix viz"

# ╔═╡ 5ae47630-5f64-11eb-39f8-654f8d277674
"""
visualize an incomplete property-material matrix.
"""
function viz_matrix(X_θ::Array{Union{Float64, Missing}, 2})
	θ = sum(.! ismissing.(X_θ)) / (n_m * n_p) # compute θ
	
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
	is = imshow(a_to_color.(X_θ'), interpolation="None")
	xlabel("gas\nadsorption\nproperties")
	ylabel("COFs")
	xticks([])
	yticks([])
	ylim([-0.5, size(X_θ')[1]-0.5])
	xlim([-0.5, size(X_θ')[2]-0.5])
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
# data structure containing low rank model hyper-parameters
struct HyperParam
	k::Int      # dimension of the latent space
	λ::Float64  # regularization parameter in loss function
end

# ╔═╡ 8152d710-5b90-11eb-39f5-45d81aa298ab
"""
fit a low rank model (lrm) to an incomplete property-material matrix.

the low rank model is:
    A ≈ M ' * P + μ * 1'
or
  X=A'≈ P ' * M + 1 * μ '
(operates on transpose inside b/c LowRankModels.jl adds offsets to columns)

# arguments
- X_θ: incomplete property-material matrix. make sure it is normalized.
- hp: hyper-parameters for the fit
- ids_obs: list of observed entries to fit the model to (don't compute from X_θ inside since we may wish to ignore a fraction of them during training)
- P₀: initial guess for P
- M₀: initial guess for M

# returns
- P: latent property vectors
- M: latent material vectors
- lrm: low rank model
- ch: convergence history

note: material bias is in the last row of M.
"""
function fit_lrm(X_θ::Array{Union{Float64, Missing}, 2}, 
		         hp::HyperParam,
		         ids_obs::Array{Tuple{Int64, Int64}, 1};
				 P₀::Union{Array{Float64, 2}, Nothing}=nothing, 
		         M₀::Union{Array{Float64, 2}, Nothing}=nothing)
	# assert we receive the transpose of the material-property matrix.
	@assert size(X_θ) == (n_p, n_m)
	
	# quadratic regularizers on the latent vectors
    rp = QuadReg(hp.λ / n_p)
    rm = QuadReg(hp.λ / n_m)
	
	# guess for the latent property and material vectors, respectively.
	#   helpful from a speed standpoint
	if isnothing(P₀) 
		P₀ = randn(hp.k + 1, n_p) 
	end
	if isnothing(M₀)
		M₀ = randn(hp.k + 1, n_m)
	end
	
	# see docs: https://github.com/madeleineudell/LowRankModels.jl
	#   low rank model is: A' ≈ X' Y
	#   for us: A  ≈ M' P
	#      ==>  A' ≈ P' M
	#   so... in LowRankModels.jl, X = P, Y = M
    lrm = GLRM(X_θ, QuadLoss(), rp, rm, hp.k + 1, 
		        obs=ids_obs, offset=true, X=P₀, Y=M₀)
#    init_svd!(glrm)
	# fit the model.
    P, M, ch = fit!(lrm)
	
	# some tests
    @assert size(P) == (hp.k + 1, n_p) # k + 1 b/c bias included.
    @assert size(M) == (hp.k + 1, n_m) # k + 1 b/c bias included.
    @assert isapprox(impute(lrm), P' * M)
    return P, M, lrm, ch
end

# ╔═╡ 21995e36-5f69-11eb-0a95-13d0136099df
"""
fit a low rank model with only a bias term for the materials.

# arguments
- X_θ: incomplete property-material matrix. make sure it is normalized.
- ids_obs: list of observed entries to fit the model to (don't compute from At inside since we may wish to ignore a fraction of them during training)
"""
function fit_bias_only_lrm(X_θ::Array{Union{Float64, Missing}, 2},
		                   ids_obs::Array{Tuple{Int64,Int64}, 1})
	hp = HyperParam(0, 0.0)
	return fit_lrm(X_θ, hp, ids_obs)
end

# ╔═╡ 3aba3150-65cd-11eb-2c51-878ef71193ac
with_terminal() do 
	@testset "fitting GLRM" begin
		# begin
		X_θ, ids_obs, ids_unobs = sim_data_collection(0.9)

		hp = HyperParam(3, 0.4)

		P₀ = randn(hp.k + 1, n_p)
		M₀ = randn(hp.k + 1, n_m)

		P, M, lrm, ch = fit_lrm(X_θ, hp, observations(X_θ), 
			P₀=P₀, M₀=M₀)
		# end
		actualP = P[1:hp.k, :] # latent vecs only, without biases
		actualM = M[1:hp.k, :] # latent vecs only, without biases
		μ = M[end, :]
		@test impute(lrm) ≈ P' * M # b/c it operates on A'
		@test impute(lrm) ≈ actualP' * actualM + ones(n_p) * μ'
		@test all(P[end, :] .≈ 1.0)
		# model is X' = A ≈ M' * P
		@test size(X_θ') == size(M' * P) == size(actualM' * actualP)
		
		# bias-only model
		P, M, lrm, ch = fit_bias_only_lrm(X_θ, observations(X_θ))
		@test size(M) == (1, n_m)
		@test size(P) == (1, n_p)
		@test impute(lrm) ≈ P' * M
	end
end

# ╔═╡ e2dd554c-5baf-11eb-1b49-654d19bedecc
md"# for evaluating the glrm"

# ╔═╡ 168d4c44-5bb3-11eb-2d16-af69f613b625
"""
compute the normalized, complete property-material matrix using the means and standard deviations of the properties that are provided (computed from train data).

# arguments
- μs: means of properties
- σs: standard deviations of properties
"""
function compute_X_normalized(μs::Array{Float64, 1}, σs::Array{Float64, 1})
	Xn = deepcopy(X)
	for p = 1:n_p
        Xn[p, :] .= (Xn[p, :] .- μs[p]) ./ σs[p]
    end
	return Xn
end

# ╔═╡ 36da7c1e-5bb1-11eb-2bc9-438dabcf9cc5
"""
compute the Spearmann rank correlation coefficient between the true values and predicted values for a given property.

# arguments
- p: the property
- X_n: the true, complete property-material matrix (normalized)
- X̂_n: the predicted property-material matrix (using normalized variables).
- ids_test: the id's of the entries comprising the test set over which to compute the Spearmann rank correlation coefficient.

# returns
- ρ: the Spearmann rank corr coeff
-
"""
function ρ_p(p::Int,
		     X_n::Array{Float64, 2},
			 X̂_n::Array{Number, 2},
		     ids_test::Array{Tuple{Int64,Int64}, 1}
		    )
	@assert size(X_n) == size(X̂_n) == (n_p, n_m)
	
	# get all test data involving property p
	ids_test_p = [(this_p, m) for (this_p, m) in ids_test if this_p == p]
	
	# true values of the property
	a = [X_n[p, m] for (p, m) in ids_test_p]
	# predicted values of the property
	â = [X̂_n[p, m] for (p, m) in ids_test_p]
	return corspearman(a, â)
end

# ╔═╡ 3df2fa8d-760b-4081-8c68-b969a65922be
with_terminal() do 
	@testset "fitting GLRM" begin
 		@test ρ_p(1, convert(Array{Float64, 2}, X), convert(Array{Number, 2}, X), [(1, 1), (1, 4), (1, 5), (3, 5)]) ≈ 1.0
	end
end

# ╔═╡ efc74f24-5ba0-11eb-2c44-6dac87ec534a
md"# hyperparam grid sweep"

# ╔═╡ effc4d2b-303e-412c-806d-fc016c88756f
# hyperparameter grid.
#   important! 
#     * rank k is outer loop
#     * reg param λ starts small, gets larger
hp_grid = [HyperParam(k, λ) for λ in 10.0 .^ range(1.0, 3.0, length=3), k = 1:3][:]
# hp_grid = [HyperParam(k, λ) for λ in 10.0 .^ range(1.0, 3.0, length=25), k = 1:15][:]

# ╔═╡ 2ff3b2f6-65db-11eb-156d-7fbcc6c79e76


# ╔═╡ 9cf75472-5ba0-11eb-371a-5bc338946b61
"""
conduct a hyper-parameter sweep.
loop over all hyper params in the hyper parameter grid, train a low rank model on train set, compute performance on validation set. return best hyper param values, judged by spearmann rank correlation coefficient.

# arguments
- hp_grid: the list of hyper params to try
- X_θ: incomplete property-material matrix (must be normalized)
- ids_train: entries that are training data
- ids_valid: entries that are validation data
- show_progress=false: shows a progress meter

# returns
- best_hp: best hyper params
- best_ρ
"""
function hyperparam_sweep(hp_grid::HPGrid, 
						  X_θ::Array{Union{Float64, Missing}, 2},
						  ids_train::Array{Tuple{Int64,Int64},1},
						  ids_valid::Array{Tuple{Int64,Int64},1};
						  show_progress::Bool=false
						 )
	# set up progress meter
	pm = Progress(length(hp_grid))
	
	# best hyper-param and Spearmann rank, ρ to return (keep over-writing)
	best_hp = HyperParam(-1, Inf)
	best_ρ = -Inf
	
	# initialize P, M
	P = randn(hp_grid[1].k + 1, n_p)
	M = randn(hp_grid[1].k + 1, n_m)
	
	# conduct sweep. r = run, hp = hyper parameter
	for (r, hp) in enumerate(hp_grid)
		# update P, M only if k changes
		if (r > 1) & hp.k != hp_grid[r-1].k
			P = randn(hp.k + 1, n_p)
			M = randn(hp.k + 1, n_m)
		end
		
		if show_progress
			update!(pm, r)
		end
	
		# train on training data
		P, M, glrm, ch = fit_lrm(X_θ, hp, ids_train, P₀=P, M₀=M)
		
		# impute missing entries
		X̂ = impute(glrm)
		
		# actual and predicted validation entries
		a = [X_θ[p, m] for (p, m) in ids_valid]
		â = [  X̂[p, m] for (p, m) in ids_valid]
		
		# compute spearman rank correl coeff on validation data
		ρ = corspearman(a, â)
		
		if ρ > best_ρ
			best_ρ = ρ
			best_hp = hp
		end
	end
	return best_hp, best_ρ
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
"""
conduct a simulation by:
1. simulating the data collection procedure which introduces missing values into the property-material matrix
2. splitting observed values into train and validation set (the remaining unobserved = test data)
3. do hyperparam sweep to find optimal hyperparams, judged by performance on validation data
4. now that we hv optimal hyper params, retrain on all observed data (train + validation) to give deployment lrm
5. evaluate deployment lrm (its imputations) on test entries
6. return (true values, pred values)

# arguments
- θ: fraction missing entries
- show_progress=false: progress bar
"""
function run_simulation(θ::Float64; show_progress::Bool=false)
	# generate incomplete property-material matrix
	X_θ, ids_obs, ids_unobs = sim_data_collection(0.9)

	# normalize rows; return mean, std of properties used in the normalization
	μs, σs = normalize!(X_θ)

	# get the observed and unobserved values
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
all_latent_vectors = hcat(res.M[1:end-1, :], res.P[1:end-1, :])

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
		scatter(p_vecs[1, p], p_vecs[2, p], edgecolor="k", color=cs[p], marker="s")
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

# ╔═╡ 2b5e6c72-86c4-11eb-0f34-47ce8229002d
function viz_prop_latent_space_some(which_props::Array{Int64, 1})
	figure()
	
	if res.hp.k > 2
		xlabel("UMAP dimension 1")
		ylabel("UMAP dimension 2")
	else
		xlabel("latent dimension 1")
		ylabel("latent dimension 2")
	end
	texts = []
	for p in which_props
		scatter(p_vecs[1, p], p_vecs[2, p], edgecolor="k", color="C5", marker="s")
		push!(texts, 
			annotate(prop_to_label[properties[p]], 
				(p_vecs[1, p], p_vecs[2, p]), 
				fontsize=10, ha="center", color="C5"
				# arrowprops=Dict(:facecolor="gray", :shrink=0.05)
			)
			)
	end
	adjustText.adjust_text(texts, force_text=0.02, force_points=3.0)
	# text(-2, 4.5, 
	# 	@sprintf("hyperparameters:\nk = %d\nλ = %.2f", res.hp.k, res.hp.λ),
	# 	ha="center", va="center", fontsize=20)
	legend(title=@sprintf("θ = %.1f\n\nk = %d\nλ = %.2f", θ, res.hp.k, res.hp.λ), loc="best")
	axvline(x=0.0, color="lightgray", zorder=0)
	axhline(y=0.0, color="lightgray", zorder=0)
	# colorbar(label=prop_to_label[properties[p]], extend="both")
	gca().set_aspect("equal", "box")
	title("map of adsorption propeties")
	tight_layout()
	savefig("prop_latent_space_few.pdf", format="pdf")
	gcf()
end

# ╔═╡ ab3a5568-5f88-11eb-373a-2f79bfce3cff
viz_prop_latent_space()

# ╔═╡ ab6a733c-86c3-11eb-0318-c19912136e3d
viz_prop_latent_space_some([12, 11, 15])

# ╔═╡ 59a72a22-5f82-11eb-1424-0913e7830bc4
function color_latent_material_space()
	prop_ids = [15, 12, 11]
	
	figs, axs = subplots(1, 3, figsize=(6.4*3, 4.8), sharey=true)
	plot_to_color = nothing
	for i = 1:3
	# scatter(p_vecs[1, p], p_vecs[2, p], marker="x", s=45, zorder=1000, color="k")
		plot_to_color = axs[i].scatter(
			m_vecs[1, :], m_vecs[2, :], c=A_n[:, prop_ids[i]], s=75, 
			vmin=-3.0, vmax=3.0, cmap="PiYG", edgecolor="k")
		
		axs[i].axvline(x=0.0, color="lightgray", zorder=0)
		axs[i].axhline(y=0.0, color="lightgray", zorder=0)
		axs[i].set_aspect("equal", "box")
		axs[i].set_title("color: " * replace(prop_to_label[properties[prop_ids[i]]], "\n" => " "))
	end
	# for colorbar to be right height
	# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph

	# if res.hp.k > 2
		axs[1].set_ylabel("UMAP dimension 2")
		for i = 1:3
			axs[i].set_xlabel("UMAP dimension 1")
		end
	# else
	# 	xlabel("latent dimension 1")
	# 	ylabel("latent dimension 2")
	# end

	# text(-3, 4.5, 
	# 	@sprintf("hyperparameters:\nk = %d\nλ = %.2f", res.hp.k, res.hp.λ),
	# 	ha="center", va="center")
	axs[3].legend(title=@sprintf("θ = %.1f, k = %d, λ = %.2f", θ, res.hp.k, res.hp.λ))
	ax = gca()
	
	divider = axes_grid1.make_axes_locatable(ax)
	cax = divider.append_axes("right", size="3%", pad="2%")
	colorbar(plot_to_color, label="standardized property value", extend="both", cax=cax)
	suptitle("map of COFs", fontsize=25)
	tight_layout()
	savefig("latent_mat_space_few.pdf", format="pdf")
	
	gcf()
end

# ╔═╡ b0619008-5f86-11eb-11b6-c7a3c4db9fd3
color_latent_material_space()

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

# ╔═╡ 8395e26e-86c2-11eb-16e6-0126c44ff298
DO_SIMS = false

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
	if ! DO_SIMS
		nb_sims = 0
	end
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
# ╟─43935790-86c3-11eb-3b21-cde10d72855b
# ╟─ae415fa4-5b82-11eb-0051-072097bb0439
# ╠═b3deec9e-5b82-11eb-0e37-abd2ac9d4b44
# ╟─bf9ed538-5b82-11eb-1198-3d35a209c5c0
# ╠═c5d42b2e-5b82-11eb-0631-35efb6b0800c
# ╟─5a74052f-dee7-4d7c-9611-c8d5c09cc778
# ╠═cd34bdd0-c427-407c-a7cd-155773b3a3c0
# ╟─6713990c-5b8d-11eb-2196-7182f36cad59
# ╠═2931005c-5b8d-11eb-2375-5dacf441be72
# ╠═44be5768-d5d6-468a-ba5b-541a7f1213a9
# ╟─1d363324-5b8f-11eb-2c78-6980a0d5f110
# ╠═2366b3cc-5b8f-11eb-07e4-61cbc97d5480
# ╠═b6a8b717-0f44-4eba-bff9-073079687dd9
# ╟─6236b296-5f60-11eb-3aa7-5188433b3906
# ╠═5ae47630-5f64-11eb-39f8-654f8d277674
# ╟─6ef474cc-5b90-11eb-2fe2-87bc7b48f5b7
# ╠═8a3c55ae-5ba4-11eb-3354-f9a8feaa7e91
# ╠═8152d710-5b90-11eb-39f5-45d81aa298ab
# ╠═21995e36-5f69-11eb-0a95-13d0136099df
# ╠═3aba3150-65cd-11eb-2c51-878ef71193ac
# ╟─e2dd554c-5baf-11eb-1b49-654d19bedecc
# ╠═168d4c44-5bb3-11eb-2d16-af69f613b625
# ╠═36da7c1e-5bb1-11eb-2bc9-438dabcf9cc5
# ╠═3df2fa8d-760b-4081-8c68-b969a65922be
# ╟─efc74f24-5ba0-11eb-2c44-6dac87ec534a
# ╠═effc4d2b-303e-412c-806d-fc016c88756f
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
# ╠═2b5e6c72-86c4-11eb-0f34-47ce8229002d
# ╠═ab3a5568-5f88-11eb-373a-2f79bfce3cff
# ╠═ab6a733c-86c3-11eb-0318-c19912136e3d
# ╠═59a72a22-5f82-11eb-1424-0913e7830bc4
# ╠═b0619008-5f86-11eb-11b6-c7a3c4db9fd3
# ╠═244ce106-65e4-11eb-080b-f52f27e435fc
# ╠═2e523504-65e4-11eb-1cbc-fd2cb39afed6
# ╟─55ee1330-6508-11eb-37d1-1973f7e077ed
# ╟─0cd6cd76-5f6e-11eb-0bf5-2f0ea61ef29b
# ╠═8395e26e-86c2-11eb-16e6-0126c44ff298
# ╠═5bbe8438-5f41-11eb-3d16-716bcb25400b
# ╠═c7aa89b0-5f93-11eb-0503-5565bba9cb86
# ╠═56bb9b5c-5f95-11eb-0d3f-97cd4b7a48a0
# ╠═0830c1de-5f9e-11eb-132a-77b3084102b2
