using CSV, DataFrames
using DifferentialEquations, Flux, DiffEqFlux, Plots, BenchmarkTools
using StochasticDiffEq
using DiffEqBase.EnsembleAnalysis
using StatsBase, Statistics
include("utils.jl")
using .Utils

data, labels = Utils.FluxLoader(norm=false)
curve_data = groupby(data, :object_id)
idx = rand(1:length(curve_data))
n_classes = size(lables)[2]

m = Utils.get_curve(curve_data, idx)
t = m[1, :]
target_data = m[2:3, :]
u0 = target_data[:, 1]
dim = length(u0)

sde_data_vars = zeros(dim, length(t)) .+ 1e-3

drift_dudt = Chain(
    Dense(dim, 20, tanh),
    # Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, n_classes),
    softmax
) #|> gpu

diffusion_dudt = Chain(Dense(dim,n_classes))

n_sde = NeuralDSDE(drift_dudt,diffusion_dudt,(t[1], t[end]),SOSRI(),saveat=t,reltol=1e-1,abstol=1e-1)
ps = Flux.params(n_sde)
cur_pred = n_sde(u0)

function predict_n_sde()
  Array(n_sde(u0))
end

function loss_n_sde(;n=5)
  samples = [predict_n_sde() for i in 1:n]
  means = reshape(mean.([[samples[i][j] for i in 1:length(samples)] for j in 1:length(samples[1])]),size(samples[1])...)
  vars = reshape(var.([[samples[i][j] for i in 1:length(samples)] for j in 1:length(samples[1])]),size(samples[1])...)
  sum(abs2,labels[idx, :] - means) + sum(abs2,sde_data_vars - vars)
end

repeated_data = Iterators.repeated((), 1000)
opt = ADAM(0.025)
losses = []

cb = function ()
	cur_pred = predict_n_sde()
	display(string("pred : ", cur_pred[1:2, 2:end]))
	display(string("target : ", target_data[1:2, 2:end]))
	pl = scatter(t[2:end], target_data[1, 2:end], label="flux_target", markersize=5, markercolor=:blue)
	scatter!(pl, t[2:end], target_data[2, 2:end], label="flux_err_target", markersize=5, markercolor=:green)
	scatter!(pl, t[2:end], cur_pred[1, 2:end], label="flux_pred", markersize=5, markercolor=:red) #, markercolor=:red)
	scatter!(pl, t[2:end], cur_pred[2, 2:end], label="flux_err_pred", markersize=5, markercolor=:orange)
	yticks!([-5:5;])
	xticks!(t[2]:t[end])
	plot!(pl, xlabel="normed_mjd", ylabel="normed_param", size=(900, 900))
end
Flux.train!(loss_n_sde, ps, repeated_data, opt, cb = cb)

end
