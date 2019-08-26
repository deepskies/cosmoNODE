using DiffEqFlux, Flux, DifferentialEquations
# f(x, theta, t) = nn(x, theta[t])

# function (n::NeuralODE)(x)
#     return neural_ode(
#     n.model,
#     x,
#     n.tspan,
#     n.solver,
#     n.kwargs ...)
# end

# function neural_ode(model, x, tspan, solver ); kwargs ...)


down = Chain(
            Conv((3, 3), 1=>64, relu, stride=1, pad=1),
            GroupNorm(64, 64),
            Conv((4, 4), 64=>64, relu, stride=2, pad=1),
            Conv((4, 4), 64=>64, stride=2, pad=1),
            )

dudt = Chain(
            Conv((3, 3), 64=>64, relu, stride=1, pad=1),
            Conv((3, 3), 64=>64, relu, stride=1, pad=1),
            )

fc = Chain(
            GroupNorm(64, 64),
            x->relu.(x),
            MeanPool((6,6)),
            x -> reshape(x, (64, bs)),
            Dense(64, 10)
            )

solver_kwargs = Dict(:reltol=>1e-3, :save_everystep=>false)
node_layer = DiffEqFlux.NeuralODE(dudt, (0.f0, 1.f0), Tsit5(), solver_kwargs)

model = Chain(
            down,
            node_layer,
            fc
            )

function loss(x, y)
    y_hat = model(x)
    return logitcrossentrpoy(y_hat, y)
end

opt = ADAM

Flux.train(loss, params(model), zip(x_train, y_train), opt)


# function odenet(z)
#     prob = ODEProblem(f, x, tspan, params)
#     return solve(prob, Tsit5())
# end
#
