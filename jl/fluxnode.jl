#=
1.




=#

module FL
    include("./fluxnet.jl")
    using .FL
    using DifferentialEquations, Flux, DiffEqFlux
    function fluxnode()
        X, Y = FL.FluxLoader()
    end

    function 
