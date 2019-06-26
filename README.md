# MOF Recommender System
### (Order not finalized) Arni Sturluson, Grant McConachie, Samuel Hough, Melanie Huynh, Cory Simon

The goal of this project is to create a recommender system for metal-organic frameworks (MOFs) in terms of adsorption of gas molecules.

The [NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials](https://adsorbents.nist.gov/isodb/index.php#home) contains numerous experimental isotherms that are used in this project.

`SparsityMatrix.ipynb` is a notebook analyzing the sparsity of the data in the NIST Database.

`HenryMatrix.ipynb` grabs the isotherm data from the NIST Database and constructs a fully formed matrix that is used for the recommender system.

`HenryFitting.ipynb` tests various methods to fit the Henry constants. Within, we use a set of 24 adsorption isotherms that we have manually fitted to compare the different methods.
