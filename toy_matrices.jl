### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 6031e0aa-65e0-11eb-1445-d38e02f9e22d
begin
	using PyPlot, Random, Statistics
	PyPlot.matplotlib.style.use("https://gist.githubusercontent.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea/raw/64a78942d3f7b4b5054902f2cee84213eaff872f/matplotlibrc")
end

# ╔═╡ 8ad981d2-65e0-11eb-28f2-dd661f0880be
function just_matrix(movies::Bool)
	
	Random.seed!(97333)
	
	norm = PyPlot.matplotlib.colors.Normalize(vmin=-1.25, vmax=1.25)
	cmap = PyPlot.matplotlib.cm.get_cmap("PiYG") # diverging
	
	# mapping adsorption properties to colors
	function a_to_color(a::Union{Float64, Missing})
		if ismissing(a)
			return (0.5, 0.5, 0.5, 1.0)
		else
			return cmap(norm(a))
		end
	end
	
	xlabels = ["properties", "movies"]
	ylabels = ["materials",  "users"]
	cbar_labels = ["value", "rating"]
	θ = 0.4
	toy_matrix = zeros(Union{Float64, Missing}, 20, 6)
	for ix = 1:size(toy_matrix)[1]
		for iy = 1:size(toy_matrix)[2]
			if rand() > θ
				toy_matrix[ix, iy] = missing
			else
				toy_matrix[ix, iy] = randn()
			end
		end
	end
	# if any row or column is all missing, fill in a random value
	for i = 1:size(toy_matrix)[1]
		if all(ismissing.(toy_matrix[i, :]))
			toy_matrix[i, rand(1:size(toy_matrix)[2])] = randn()
		end
		if all(.! ismissing.(toy_matrix[i, :]))
			toy_matrix[i, rand(1:size(toy_matrix)[2])] = missing
		end
	end
	if movies
		toy_matrix = toy_matrix'
	end
	# normalize cols
	for j = 1:size(toy_matrix)[2]
		ids_observed = .! ismissing.(toy_matrix[:, j])
		μ = mean(toy_matrix[ids_observed, j])
		σ = std(toy_matrix[ids_observed, j])
		toy_matrix[:, j] = (toy_matrix[:, j] .- μ) / σ
	end

	if movies
		figure(figsize=(6, 2.6))
		
	else
		figure(figsize=(2.6, 5))
	end
	ax = gca()
	is = imshow(a_to_color.(toy_matrix))
	ylim([-0.5, size(toy_matrix)[1]-0.5])
	xlim([-0.5, size(toy_matrix)[2]-0.5])
	xticks([])
	yticks([])

	colorbar(PyPlot.matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.5, ticks=[0], label=movies ? "rating" : "value")
	if movies
		ylabel("← products →")
		xlabel("← users →")
		tight_layout()
		savefig("um_toy_matrix.pdf", format="pdf")
	else
		xlabel("← properties →")
		ylabel("← materials →")
		
		tight_layout()
		savefig("mp_toy_matrix.pdf", format="pdf")
	end
	gcf()
end

# ╔═╡ 975d209e-65e0-11eb-3220-fdf3e66b1743
just_matrix(true)

# ╔═╡ 3e556e5a-6d64-11eb-100b-29093f91e0ee
just_matrix(false)

# ╔═╡ Cell order:
# ╠═6031e0aa-65e0-11eb-1445-d38e02f9e22d
# ╠═8ad981d2-65e0-11eb-28f2-dd661f0880be
# ╠═975d209e-65e0-11eb-3220-fdf3e66b1743
# ╠═3e556e5a-6d64-11eb-100b-29093f91e0ee
