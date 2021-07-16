# material recommendation system

this repo contains all data and code to reproduce all plots in our article:

:evergreen_tree: A. Sturluson, A. Raza, G. McConachie, D. Siderius, X. Fern, C. Simon. "A recommendation system to predict missing adsorption properties of nanoporous materials." [ChemRxiv preprint](https://chemrxiv.org/engage/chemrxiv/article-details/60c757ab337d6cfa42e2906d).

## querying AiiDA for data regarding simulated gas adsorption in COFs

the Python code to query the COF adsorption data via AiiDA (see [paper](https://pubs.acs.org/doi/10.1021/acscentsci.0c00988) and [Materials Cloud](https://archive.materialscloud.org/record/2021.100) v9) is in the Jupyter Notebook `AiiDAQuery.ipynb`, while `AiiDAPrimer.ipynb` gives a primer on AiiDA. 

the result of the query-- the simulated adsorption data for this project-- is contained in `aiida_ads_data_june21.csv` and read into `cof_rec_sys.jl`.

`cof-frameworks.csv` contains information about the COF structures. particularly, it links the COF IDs in `aiida_ads_data_oct20.csv` with the common name of the COFs.

## training the low rank models

`cof_rec_sys.jl` is a Pluto notebook with Julia code to train and evaluate the low rank models of the COF--gas-adsorption-property matrix and visualize the results. use Julia v1.6 and Pluto 0.15.1. Pluto will automatically install the required Julia packages for you. see [here](https://github.com/fonsp/Pluto.jl) and [here](https://julialang.org/) for more information on Pluto and Julia, respectively.

the Pluto notebook is commented and functions annotated to understand it.

## further visualizations

we used the Seaborn library in Python to make further visualizations. `seaborn_stuff.ipynb` reads in data written from `cof_rec_sys.jl` to, for example, visualize the distribution of the hyper-parameters among the low rank models.
