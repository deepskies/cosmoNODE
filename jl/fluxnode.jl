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

	return
        # X, Y = FL.FluxLoader()
    end

    function
