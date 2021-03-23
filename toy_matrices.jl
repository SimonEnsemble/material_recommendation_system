### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 6031e0aa-65e0-11eb-1445-d38e02f9e22d
begin
	using PyPlot, Random, Statistics
	PyPlot.matplotlib.style.use("https://gist.githubusercontent.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea/raw/64a78942d3f7b4b5054902f2cee84213eaff872f/matplotlibrc")
end

# ╔═╡ f389a492-8b87-11eb-2a51-87c9a84c8dd0
xkcd()

# ╔═╡ 3dfc943a-8b88-11eb-0c77-954777f469f3
begin
	
	norm = PyPlot.matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
	cmap = PyPlot.matplotlib.cm.get_cmap("terrain") # diverging
	
	# mapping adsorption properties to colors
	function a_to_color(a::Union{Float64, Missing})
		if ismissing(a)
			return (1.0, 1.0, 1.0, 1.0)
		else
			return cmap(norm(a))
		end
	end
	
	θ = 0.7
	toy_matrix = zeros(Union{Float64, Missing}, 5, 3)
	for ix = 1:size(toy_matrix)[1]
		for iy = 1:size(toy_matrix)[2]
			if rand() > θ
				toy_matrix[ix, iy] = missing
			else
				toy_matrix[ix, iy] = rand()
			end
		end
	end
	# if any row or column is all missing, fill in a random value
	for i = 1:size(toy_matrix)[1]
		if all(ismissing.(toy_matrix[i, :]))
			toy_matrix[i, rand(1:size(toy_matrix)[2])] = rand()
		end
		if all(.! ismissing.(toy_matrix[i, :]))
			toy_matrix[i, rand(1:size(toy_matrix)[2])] = missing
		end
	end
	
	figure()
	n_row, n_col = size(toy_matrix)
	is = imshow(a_to_color.(toy_matrix))
	for i = 1:n_row
		for j = 1:n_col
			if ismissing(toy_matrix[i, j])
				text(j - 1, i - 1, "?", fontsize=35,va="center", ha="center")
			end
		end
	end
	# rows
	for i = 0:n_row
		plot([0, n_col] .-0.5, [i, i].-0.5, color="k")
	end
	for i = 0:n_col
		plot([i, i].-0.5, [0, n_row].-0.5, color="k")
	end
	gca().set_aspect("equal", "box")
	ylim([-0.55, size(toy_matrix)[1]-0.45])
	xlim([-0.55, size(toy_matrix)[2]-0.45])
	xticks([])
	yticks([])
	tight_layout()
	savefig("toc_image.pdf", format="pdf")
	gcf()
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

# ╔═╡ fac4ef18-7f08-11eb-0dfb-a99468b3340c
function very_toy()
	
	Random.seed!(97336)
	
	norm = PyPlot.matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
	cmap = PyPlot.matplotlib.cm.get_cmap("plasma") # diverging
	
	# mapping adsorption properties to colors
	function a_to_color(a::Union{Float64, Missing})
		if ismissing(a)
			return (1.0, 1.0, 1.0, 1.0)
		else
			return cmap(norm(a))
		end
	end
	
	θ = 0.7
	toy_matrix = zeros(Union{Float64, Missing}, 5, 3)
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
			toy_matrix[i, rand(1:size(toy_matrix)[2])] = rand()
		end
		if all(.! ismissing.(toy_matrix[i, :]))
			toy_matrix[i, rand(1:size(toy_matrix)[2])] = missing
		end
	end

	figure(figsize=(2.6, 5))
	ax = gca()
	is = imshow(a_to_color.(toy_matrix))
	hlines([i - 0.5 for i = 0:size(toy_matrix)[1]], 
		-0.5, size(toy_matrix)[2]-0.5, 
		color="k", clip_on=false)
	vlines([i - 0.5 for i = 0:size(toy_matrix)[2]], 
		-0.5, size(toy_matrix)[1]-0.5, 
		color="k", clip_on=false)
	ylim([-0.5, size(toy_matrix)[1]-0.5])
	xlim([-0.5, size(toy_matrix)[2]-0.5])
	xticks([])
	yticks([])

	colorbar(PyPlot.matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.5, ticks=[0], label="value")

	tight_layout()
	savefig("very_toy_matrix.png", format="png", dpi=400)
	gcf()
end

# ╔═╡ 3d88200e-7f09-11eb-179a-afe22a3461f3
very_toy()

# ╔═╡ 93d6d2dc-8b8b-11eb-3412-49020bd98009
begin
	figure()
	scatter(0.95*(2*rand(n_row).-1), 0.95*(2* rand(n_row).-1), s=100)
	plot([0, 0], [-1, 1], color="k", zorder=0)
	plot([-1, 1], [0, 0], color="k", zorder=0)
	xticks([])
	yticks([])
	tight_layout()
	savefig("latent_toc.pdf", format="pdf")
	gcf()
end

# ╔═╡ Cell order:
# ╠═6031e0aa-65e0-11eb-1445-d38e02f9e22d
# ╠═8ad981d2-65e0-11eb-28f2-dd661f0880be
# ╠═975d209e-65e0-11eb-3220-fdf3e66b1743
# ╠═3e556e5a-6d64-11eb-100b-29093f91e0ee
# ╠═fac4ef18-7f08-11eb-0dfb-a99468b3340c
# ╠═3d88200e-7f09-11eb-179a-afe22a3461f3
# ╠═f389a492-8b87-11eb-2a51-87c9a84c8dd0
# ╠═3dfc943a-8b88-11eb-0c77-954777f469f3
# ╠═93d6d2dc-8b8b-11eb-3412-49020bd98009
