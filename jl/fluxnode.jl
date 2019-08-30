#=
1.




=#

module FluxNode
    include("./fluxnet.jl")
    using .FL
    using DifferentialEquations, Flux, DiffEqFlux, CSV

    function fluxnode()
	df = CSV.read("../demos/data/training_set.csv")
	groups = groupby(df, [:object_id, :passband])
	g = groups[1]
	df_lc = g[:, [:
	band_data = convert(Matrix, g)
	
	return
        # X, Y = FL.FluxLoader()
    end

    function
