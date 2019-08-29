using DifferentialEquations, Flux, Plots, DiffEqFlux

u0 = Float32[0.0]

datasize = 1000
tspan = (0.0f0, Float32(8π)) 

function trueODEfunc(du, u, p, t)
	du .= cos.(t)
end

t = range(tspan[1], tspan[2], length=datasize)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat=t))

dudt = Chain(
	     Dense(1, 50, tanh),
	     Dense(50, 1))

ps = Flux.params(dudt)
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)

pred = n_ode(u0)
scatter(t, ode_data[1, :], label="data")
scatter!(t, Flux.data(pred[1,:]), label="prediction")

function predict_n_ode()
	n_ode(u0)
end
loss_n_ode() = sum(abs2, ode_data .-predict_n_ode())

data = Iterators.repeated((), 1000)
opt = ADAM(0.1)

cb = function ()
	display(loss_n_ode())
	cur_pred = Flux.data(predict_n_ode())
	pl = scatter(t, ode_data[1, :], label="data")
	scatter!(pl, t, cur_pred[1, :], label="prediction")
	display(plot(pl))
end

cb()

Flux.train!(loss_n_ode, ps, data, opt, cb=cb)

