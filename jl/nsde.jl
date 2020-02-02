using CSV, DataFrames
using DifferentialEquations, Flux, DiffEqFlux, Plots, BenchmarkTools
using StochasticDiffEq
using DiffEqBase.EnsembleAnalysis
using StatsBase, Statistics
include("utils.jl")
using .Utils

data, labels = Utils.FluxLoader(norm=false)
curves = groupby(data, :object_id)
idx = 30
k = rand(keys(curves))
typeof(k)

m = Utils.get_curve(curves, idx)
t = m[1, :]
target_data = m[2:end, :]
u0 = target_data[:, 1]
dim = length(u0)

sde_data_vars = zeros(dim, length(t))

drift_dudt = Chain(
    Dense(dim, 20, tanh),
    # Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, dim)
) #|> gpu

diffusion_dudt = Chain(Dense(dim,dim))

n_sde = NeuralDSDE(drift_dudt,diffusion_dudt,(t[1], t[end]),SOSRI(),saveat=t,reltol=1e-1,abstol=1e-1)
ps = Flux.params(n_sde)
pred = n_sde(u0)

function predict_n_sde()
  Array(n_sde(u0))
end

function loss_n_sde(;n=10)
  samples = [predict_n_sde() for i in 1:n]
  means = reshape(mean.([[samples[i][j] for i in 1:length(samples)] for j in 1:length(samples[1])]),size(samples[1])...)
  vars = reshape(var.([[samples[i][j] for i in 1:length(samples)] for j in 1:length(samples[1])]),size(samples[1])...)
  sum(abs2,target_data - means) + sum(abs2,sde_data_vars - vars)
end

repeated_data = Iterators.repeated((), 1000)
opt = ADAM(0.025)

losses = []

cb = function ()
  cur_pred = predict_n_sde()
  display(string("target data shape: ", size(target_data)))
  display(string("pred shape: ", size(cur_pred)))
  pl = scatter(t[2:end], target_data[:, 2:end]', label="data", markershape=:star, markersize=5) #, markercolor=:blue)
  scatter!(pl, t[2:end], cur_pred[:, :]', label="prediction", markersize=5) #, markercolor=:red)
  yticks!([-5:5;])
  xticks!(t[1]:t[end])

  display(plot(pl, size=(900, 900)))
end


Flux.train!(loss_n_sde, ps, repeated_data, opt, cb = cb)

end
