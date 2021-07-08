### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 92502b2a-7f83-11eb-152b-f10d7015d5cc
begin
	using LowRankModels, CSV, DataFrames, PyPlot, Statistics, Distributions, StatsBase, Printf, UMAP, PyCall, ProgressMeter, Random, Test, PlutoUI
	
	using ScikitLearn.CrossValidation: train_test_split
	
	PyPlot.matplotlib.style.use("https://gist.githubusercontent.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea/raw/64a78942d3f7b4b5054902f2cee84213eaff872f/matplotlibrc")
	
	adjustText = pyimport("adjustText")
	sbn = pyimport("seaborn")
	axes_grid1 = pyimport("mpl_toolkits.axes_grid1")
	
	const my_seed = 97330
	Random.seed!(my_seed);
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
function hyperparam_sweep(hp_grid::Array{HyperParam, 1}, 
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
		if (r > 1) && hp.k != hp_grid[r-1].k
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

# ╔═╡ 09b74408-5baa-11eb-3735-e9333756f565
struct Result
	# the data
	X_θ::Array{Union{Float64, Missing}, 2}
	
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
	X_θ, ids_obs, ids_test = sim_data_collection(θ)
	
	# normalize rows; return mean, std of properties used in the normalization
	μs, σs = normalize!(X_θ)

	# split the observed values into train and validation
	ids_train, ids_valid = train_test_split(ids_obs, test_size=0.2, 
		                                    shuffle=true, 
		                                    random_state=floor(Int, my_seed + 100 * θ))

	#   hyper-parameter sweep using train/valid data to find optimal (k, λ) hyperparam
	hp, _ = hyperparam_sweep(hp_grid, X_θ, ids_train, ids_valid, 
		                     show_progress=show_progress)

	# train lrm with all observed data to give deployment lrm
	P, M, lrm, ch = fit_lrm(X_θ, hp, ids_obs)
	Gb, Mb, lrmb, chb = fit_bias_only_lrm(X_θ, ids_obs) # bias-only
	
	# imputed matrices
	X̂  = impute(lrm)
	X̂b = impute(lrmb) # bias-only

	# compute true imputed matrix using our normalization scheme
	X̂_true = compute_X_normalized(μs, σs)

	# compute performance on test dta
	a  = [X̂_true[p, m] for (p, m) in ids_test]
	â  = [     X̂[p, m] for (p, m) in ids_test]
	âb = [    X̂b[p, m] for (p, m) in ids_test]
	
	# get spearman ranks for gases
	ρp  = [ρ_p(p, X̂_true, X̂,  ids_test) for p = 1:n_p]
	ρpb = [ρ_p(p, X̂_true, X̂b, ids_test) for p = 1:n_p]
	
	return Result(X_θ, 
		          M, P, hp,
		          a, â, âb, 
		          ρp, ρpb)
end

# ╔═╡ 5d38b414-5f41-11eb-14b9-73a9007fc263
md"# $\theta=0.4$ example"

# ╔═╡ 128a9fa0-808b-11eb-22f0-afd1dd30593c
θ = 0.4

# ╔═╡ bdb51f2e-65d8-11eb-1063-614c91a95e6e


# ╔═╡ 4f81a520-5f6d-11eb-1960-9918ca4f25e9
res = run_simulation(0.4, show_progress=true)

# ╔═╡ 68fe93ae-5f6e-11eb-012a-81378cd15b41
viz_matrix(res.X_θ)

# ╔═╡ 53585188-5f6f-11eb-0fc0-abbd20ee33fe
begin
	# parity plot
	δ = 0.1
	
	figure()
	hexbin(clamp.(res.a, -6.0, 6.0),
		   clamp.(res.â, -6.0, 6.0),
		   mincnt=1, bins="log")
	xlabel(L"true $A_{mp}$ (standardized)")
	ylabel(L"predicted $A_{mp}$ (standardized)")
	xlim([-6 - δ, 6 + δ])
	ylim([-6 - δ, 6 + δ])
	plot([-6, 6], [-6, 6], 
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

# ╔═╡ b171c70f-30d5-4e31-bdac-ef7655ac19b3
begin
	figure()
	hist(μ)
	ylabel("# COFs")
	xlabel(L"material bias $\mu_m$")
	tight_layout()
	savefig("distn_of_material_biases.pdf", format="pdf")
	gcf()
end

# ╔═╡ ce346a40-667c-11eb-03d3-eb7c4510ff26
df[ids_sort, [:cof, :Name]][1:3, :]

# ╔═╡ 1568fe16-667e-11eb-0ecc-bfd712234906
# bottom materals
df[ids_sort, [:cof, :Name]][end-2:end, :]

# ╔═╡ b0560c02-5f80-11eb-338b-c9cc48b741af
md"### learn latent space

of materials and properties together.
"

# ╔═╡ ba8ce81e-5f80-11eb-3e39-f942cb6d0d1f
begin
	# stack together latent material vectors to prepare for dim reduction
	all_latent_vectors = hcat(res.M[1:end-1, :], res.P[1:end-1, :])
	@assert size(all_latent_vectors) == (res.hp.k, n_m+n_p)
	all_latent_vectors
end

# ╔═╡ 227a2198-6d29-405a-9df8-54491c08b044

# minimize ||A - XY||^2
function fit_pca(m,n,k)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	loss = QuadLoss()
	r = ZeroReg()
	pca_model = GLRM(A, QuadLoss(), ZeroReg(), ZeroReg(), 2)
	X,Y,ch = fit!(pca_model)
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

# ╔═╡ c6caaa48-5f7f-11eb-3853-fdffcd51b2d5
begin
	# dim reduction of mat and prop vecs to 2D
	m_vecs = res.M[1:end-1, :] # initialize so works if already 2D.
	p_vecs = res.P[1:end-1, :]
	if res.hp.k > 2
		# input: (a column-major matrix of shape (n_features, n_samples))
		pca_model = GLRM(all_latent_vectors, QuadLoss(), ZeroReg(), ZeroReg(), 2)
		fit!(pca_model)
		latent_space = pca_model.Y
		# latent_space = umap(all_latent_vectors, 2)
		
		m_vecs = latent_space[:, 1:n_m]
		p_vecs = latent_space[:, (n_m+1):end]
	end
	
	@assert size(p_vecs) == (2, n_p)
	@assert size(m_vecs) == (2, n_m)
end

# ╔═╡ a1a20abe-62bf-4a02-9e48-f45aa350ff0e
# reconstruction error, RMSE:
sqrt(mean((all_latent_vectors .- impute(pca_model)) .^ 2))

# ╔═╡ 8024beae-5f88-11eb-3e97-b7afbbbc6f5c
function viz_prop_latent_space()
	cs = sbn.color_palette("husl", 16)
	
	figure(figsize=(8, 8))
	
	if res.hp.k > 2
		xlabel("principal component 1")
		ylabel("principal component 2")
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
		xlabel("principal component 1")
		ylabel("principal component 2")
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
begin
	prop_ids_we_love = [11, 13, 14]
	viz_prop_latent_space_some(prop_ids_we_love)
end

# ╔═╡ 59a72a22-5f82-11eb-1424-0913e7830bc4
function color_latent_material_space()
	X_n = compute_X_normalized(mean(X, dims=2)[:], std(X, dims=2)[:])
	
	
	figs, axs = subplots(1, 3, figsize=(6.4*1.4, 4.8), sharey=true)
	plot_to_color = nothing
	for i = 1:3
	# scatter(p_vecs[1, p], p_vecs[2, p], marker="x", s=45, zorder=1000, color="k")
		plot_to_color = axs[i].scatter(
			m_vecs[1, :], m_vecs[2, :], c=X_n[prop_ids_we_love[i], :], s=75, 
			vmin=-3.0, vmax=3.0, cmap="PiYG", edgecolor="k")
		
		axs[i].axvline(x=0.0, color="lightgray", zorder=0)
		axs[i].axhline(y=0.0, color="lightgray", zorder=0)
		axs[i].set_aspect("equal", "box")
		axs[i].set_title("color:\n" * replace(prop_to_label[properties[prop_ids_we_love[i]]], "\n" => " "))
	end
	# for colorbar to be right height
	# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph

	# if res.hp.k > 2
		axs[1].set_ylabel("principal component 2")
		for i = 1:3
			axs[i].set_xlabel("principal component 1")
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
	cax = divider.append_axes("right", size="7%", pad="4%")
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
	X_n = compute_X_normalized(mean(X, dims=2)[:], std(X, dims=2)[:])
	
	fig, axs = subplots(4, 4, sharex=true, sharey=true, figsize=(13, 11))
	p = 0
	for i = 1:4
		for j = 1:4
			p += 1
			da_plot = axs[i, j].scatter(m_vecs[1, :], m_vecs[2, :], c=X_n[p, :], s=25, 
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
			axs[i, j].set_xlabel("PC1")
			axs[i, j].set_ylabel("PC2")
			axs[i, j].axvline(x=0.0, color="lightgray", zorder=0)
			axs[i, j].axhline(y=0.0, color="lightgray", zorder=0)
		end
	end
	suptitle("COF map colored by adsorption properties", fontsize=23)
	tight_layout()
	savefig("latent_mat_space.pdf", format="pdf", bbox_inches="tight")
	gcf()
end

# ╔═╡ 2e523504-65e4-11eb-1cbc-fd2cb39afed6
color_latent_material_space_all()

# ╔═╡ 55ee1330-6508-11eb-37d1-1973f7e077ed
md"todo: color by void fraction etc."

# ╔═╡ 0cd6cd76-5f6e-11eb-0bf5-2f0ea61ef29b
md"# loop over θs
change the fraction of observed entries, see how performance depends on sparsity.
"

# ╔═╡ 8395e26e-86c2-11eb-16e6-0126c44ff298
DO_SIMS = true

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
end

# ╔═╡ 3800cb64-8419-478b-b88b-f628bcc843f9
begin
	nb_sims = 3
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

# ╔═╡ 173f8c5d-18cc-4fd2-a0f8-d6f387092da6
md"
# bootstrapping
"

# ╔═╡ 720186cc-dd8c-4ed8-92d5-c881174294e9
function run_bootstrapping(θ::Float64, 
		                   hp::HyperParam,
						   nb_bs::Int)
	# generate incomplete property-material matrix
	X_θ, ids_obs, ids_test = sim_data_collection(θ)
	
	# normalize rows; return mean, std of properties used in the normalization
	μs, σs = normalize!(X_θ)
	
	# compute true matrix using our normalization scheme
	X̂_true = compute_X_normalized(μs, σs)
	
	# the true values in the test set
	a  = [X̂_true[p, m] for (p, m) in ids_test]

	for b = 1:nb_bs
		# select bootstrap sample
		ids_bs = sample(ids_obs, length(ids_obs), replace=true)
		# train lrm on bootstrap sample
		P, M, lrm, ch = fit_lrm(X_θ, hp, ids_bs)
		# impute
		X̂ = impute(lrm)
		# the predicted properties in the test set
		â  = [X̂[p, m] for (p, m) in ids_test]
	end
end

# ╔═╡ ff0b78b4-ab6a-436d-8a81-f0bb76c3f67e
run_bootstrapping(0.4, res.hp, 10)

# ╔═╡ de4245dc-f9e4-42a5-958a-dd4235656ea7
res.hp

# ╔═╡ 07b5f0da-510f-428b-bcc9-d0d6d0b35689
θ

# ╔═╡ 050d0fd2-e138-4c0c-9edb-241059a39e12
res.X_θ

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LowRankModels = "15d4e49f-4837-5ea3-a885-5b28bfa376dc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ScikitLearn = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
UMAP = "c4f8c510-2410-5be4-91d7-4fbaeb39457e"

[compat]
CSV = "~0.8.5"
DataFrames = "~0.21.8"
Distributions = "~0.23.12"
LowRankModels = "~1.1.1"
PlutoUI = "~0.7.1"
ProgressMeter = "~1.7.1"
PyCall = "~1.92.3"
PyPlot = "~2.9.0"
ScikitLearn = "~0.6.4"
StatsBase = "~0.33.8"
UMAP = "~0.1.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "f87e559f87a45bece9c9ed97458d3afe98b1ebb9"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.1.0"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "045ff5e1bc8c6fb1ecb28694abba0a0d55b5f4f5"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.17"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "JSON", "Missings", "Printf", "Statistics", "StructTypes", "Unicode"]
git-tree-sha1 = "2ac27f59196a68070e132b25713f9a5bbc5fa0d2"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.8.3"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dc7dedc2c2aa9faf59a55c622760a25cbefbe941"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.31.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["CategoricalArrays", "Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "Missings", "PooledArrays", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ecd850f3d2b815431104252575e7307256121548"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "0.21.8"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "214c3fcac57755cfda163d91c58893a8723f93e9"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.0.2"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "abe4ad222b26af3337262b8afb28fab8d215e9f8"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.3"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "501c11d708917ca09ce357bed163dbaf0f30229f"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.23.12"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "502b3de6039d5b78c76118423858d981349f3823"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.9.7"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "f6f80c8f934efd49a286bb5315360be66956dfc4"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "e2af66012e08966366a43251e1fd421522908be6"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.18"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LightGraphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "432428df5f360964040ed60418dd5601ecd240b6"
uuid = "093fc24a-ae57-5d10-9952-331d41423f4d"
version = "1.3.5"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LowRankModels]]
deps = ["Arpack", "DataFrames", "LinearAlgebra", "NMF", "Optim", "Printf", "Random", "Requires", "ScikitLearnBase", "SharedArrays", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "cc10bb134a2eb9e6f22d10fa1bba2b3a97c2b152"
uuid = "15d4e49f-4837-5ea3-a885-5b28bfa376dc"
version = "1.1.1"

[[LsqFit]]
deps = ["Distributions", "ForwardDiff", "LinearAlgebra", "NLSolversBase", "OptimBase", "Random", "StatsBase"]
git-tree-sha1 = "b32b5549461fcb93bce223e264d4a7ef0c9923fd"
uuid = "2fda8390-95c7-5789-9bda-21331edee243"
version = "0.11.0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f8c673ccc215eb50fcadb285f522420e29e69e1c"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "0.4.5"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50608f411a1e178e0129eab4110bd56efd08816f"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.0"

[[NMF]]
deps = ["LinearAlgebra", "Printf", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "e5a5f9a6966bd0781dce72bb8bc770b68431fcb6"
uuid = "6ef6ca0d-6ad7-5ff6-b225-e928bfa0a386"
version = "0.4.1"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighborDescent]]
deps = ["DataStructures", "Distances", "LightGraphs", "Random", "Reexport", "SparseArrays"]
git-tree-sha1 = "410580927bc16e156e5481d9318b8ca177c30f1b"
uuid = "dd2c4c9e-a32f-5b2f-b342-08c2f244fce8"
version = "0.3.4"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d34366a3abc25c41f88820762ef7dfdfe9306711"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.3.0"

[[OptimBase]]
deps = ["NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "9cb1fee807b599b5f803809e85c81b582d2009d6"
uuid = "87e2bd06-a317-5318-96d9-3ecbac512eee"
version = "2.0.2"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "95a4038d1011dfdbde7cecd2ad0ac411e53ab1bc"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.10.1"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "c8abc88faa3f7a3950832ac5d6e690881590d6dc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "Logging", "Markdown", "Random", "Suppressor"]
git-tree-sha1 = "45ce174d36d3931cd4e37a47f93e07d1455f038d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.1"

[[PooledArrays]]
deps = ["DataAPI"]
git-tree-sha1 = "b1333d4eced1826e15adbdf01a4ecaccca9d353c"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "0.5.3"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "169bb8ea6b1b143c5cf57df6d34d022a7b60c6db"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.3"

[[PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "67dde2482fe1a72ef62ed93f8c239f947638e5a2"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.9.0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
deps = ["Pkg"]
git-tree-sha1 = "7b1d07f411bc8ddb7977ec7f377b97b158514fe0"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "0.2.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[ScikitLearn]]
deps = ["Compat", "Conda", "DataFrames", "Distributed", "IterTools", "LinearAlgebra", "MacroTools", "Parameters", "Printf", "PyCall", "Random", "ScikitLearnBase", "SparseArrays", "StatsBase", "VersionParsing"]
git-tree-sha1 = "ccb822ff4222fcf6ff43bbdbd7b80332690f168e"
uuid = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
version = "0.6.4"

[[ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ffae887d0f0222a19c406a11c3831776d1383e3d"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.3"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures", "Random", "Test"]
git-tree-sha1 = "03f5898c9959f8115e30bc7226ada7d0df554ddd"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "0.3.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["OpenSpecFun_jll"]
git-tree-sha1 = "d8d8b8a9f4119829410ecd706da4cc8594a1e020"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "0.10.3"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "2740ea27b66a41f9d213561a04573da5d3823d4b"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.2.5"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "da4cf579416c81994afd6322365d00916c79b8ae"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "0.12.5"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2f6792d523d7448bbe2fec99eca9218f06cc746d"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.8"

[[StatsFuns]]
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "e36adc471280e8b346ea24c5c87ba0571204be7a"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.7.2"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "8ed4a3ea724dac32670b062be3ef1c1de6773ae8"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.4.4"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UMAP]]
deps = ["Arpack", "Distances", "LinearAlgebra", "LsqFit", "NearestNeighborDescent", "Random", "SparseArrays"]
git-tree-sha1 = "c96f3a85e8d429129714a1363e622a4cb9936c79"
uuid = "c4f8c510-2410-5be4-91d7-4fbaeb39457e"
version = "0.1.8"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═92502b2a-7f83-11eb-152b-f10d7015d5cc
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
# ╟─efc74f24-5ba0-11eb-2c44-6dac87ec534a
# ╠═effc4d2b-303e-412c-806d-fc016c88756f
# ╟─2ff3b2f6-65db-11eb-156d-7fbcc6c79e76
# ╠═9cf75472-5ba0-11eb-371a-5bc338946b61
# ╟─2009751a-5bae-11eb-158f-a3d9cb98fe24
# ╠═168d4c44-5bb3-11eb-2d16-af69f613b625
# ╠═36da7c1e-5bb1-11eb-2bc9-438dabcf9cc5
# ╠═3df2fa8d-760b-4081-8c68-b969a65922be
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
# ╠═b171c70f-30d5-4e31-bdac-ef7655ac19b3
# ╠═ce346a40-667c-11eb-03d3-eb7c4510ff26
# ╠═1568fe16-667e-11eb-0ecc-bfd712234906
# ╟─b0560c02-5f80-11eb-338b-c9cc48b741af
# ╠═ba8ce81e-5f80-11eb-3e39-f942cb6d0d1f
# ╠═227a2198-6d29-405a-9df8-54491c08b044
# ╠═c6caaa48-5f7f-11eb-3853-fdffcd51b2d5
# ╠═a1a20abe-62bf-4a02-9e48-f45aa350ff0e
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
# ╠═3800cb64-8419-478b-b88b-f628bcc843f9
# ╠═c7aa89b0-5f93-11eb-0503-5565bba9cb86
# ╠═56bb9b5c-5f95-11eb-0d3f-97cd4b7a48a0
# ╠═0830c1de-5f9e-11eb-132a-77b3084102b2
# ╟─173f8c5d-18cc-4fd2-a0f8-d6f387092da6
# ╠═720186cc-dd8c-4ed8-92d5-c881174294e9
# ╠═ff0b78b4-ab6a-436d-8a81-f0bb76c3f67e
# ╠═de4245dc-f9e4-42a5-958a-dd4235656ea7
# ╠═07b5f0da-510f-428b-bcc9-d0d6d0b35689
# ╠═050d0fd2-e138-4c0c-9edb-241059a39e12
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
