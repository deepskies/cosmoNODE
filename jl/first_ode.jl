using DifferentialEquations
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)
sol = solve(prob,Tsit5())
using Plots
plot(sol)


using Flux, DiffEqFlux

p = param([2.2, 1.0, 2.0, 0.4])
params = Flux.Params([p])

function predict_rd()
    diffeq_rd(p, prob, Tsit5(), saveat=0.1)[1,:]
end

loss_rd() = sum(abs2, x-1, for x in predict_rd())

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
    display(loss_rd())
    display(plot(solve(remake(prob, p=Flux.data(p)), Tsit5(), saveat=0.1), ylim=(0, 6)))
end

cb()

Flux.train!(loss_rd, params, data, opt, cb=cb)
