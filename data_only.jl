### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 8f99766c-8c00-11eb-1120-9f07974b2d89
using CSV, DataFrames

# ╔═╡ 3ef083be-8c01-11eb-2719-67cba60a3fe6
md"# just the data matrix

when citing the source of this data, cite DOIs: 10.1021/acscentsci.9b00619, 10.1021/acscentsci.0c00988
"

# ╔═╡ 9af70fb2-8c00-11eb-2fdb-25c4a74e6601
begin
	df = CSV.read("aiida_ads_data_oct20.csv", DataFrame)
	df[:, :cof] = map(c -> split(c, ".cif")[1], df[:, :cof])
	
	for henry_col in Symbol.(["h2o_henry", "h2s_henry", "xe_henry", "kr_henry"])
		# convert units from mol/(kg-Pa) to mol/(kg-bar)
		df[!, henry_col] = 1e5 * df[:, henry_col]
		# log-10 transform
		df[!, henry_col] = log10.(df[:, henry_col])
	end
	
	df_names = CSV.read("cof-frameworks.csv", DataFrame)
	rename!(df_names, Symbol("CURATED-COFs ID") => :cof)
	df_names = df_names[:, [:cof, :Name]]

	df = leftjoin(df, df_names, on=:cof)

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
	
	df = df[:, properties]
	df # the matrix
end

# ╔═╡ Cell order:
# ╠═8f99766c-8c00-11eb-1120-9f07974b2d89
# ╟─3ef083be-8c01-11eb-2719-67cba60a3fe6
# ╠═9af70fb2-8c00-11eb-2fdb-25c4a74e6601
