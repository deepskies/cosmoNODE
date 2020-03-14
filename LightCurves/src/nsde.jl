using CSV, DataFrames
using DifferentialEquations, Flux, DiffEqFlux, Plots, BenchmarkTools
using StochasticDiffEq
using DiffEqBase.EnsembleAnalysis
using StatsBase, Statistics
include("utils.jl")
using .Utils

function view_subset(df, start_frac = 0.50, end_frac = 1, stride_amt=2)
	len = size(df)[1]
	view_start = Int(round(len * start_frac))
	view_end = Int(round(len * end_frac))
	subset = view(df, view_start:stride_amt:view_end, :)
	return subset
end

df, labels = Utils.FluxLoader(norm=false)
curve_data = groupby(df, :object_id)
idx = rand(1:length(curve_data))
# m = copy(view_subset(curve_data[idx]))
m = copy(curve_data[idx])

err = copy(m.flux_err)
t = copy((m.mjd .- m.mjd[1]) ./ 1e3)
tspan = (t[1], t[end])

target_data = Matrix(select!(m, :flux))'
u0 = target_data[:, 1]
dim = length(u0)

sde_data_vars = zeros(dim, length(t))

drift_dudt = Chain(
	Dense(dim, 16, tanh),
	Dense(16, dim)
) #|> gpu

diffusion_dudt = Chain(Dense(dim,dim))

n_sde = NeuralDSDE(drift_dudt,diffusion_dudt,tspan,SOSRI(),saveat=t,reltol=1e-3,abstol=err)
n_sde.p
pred = n_sde(u0, n_sde.p)
drift_(u,p,t) = drift_dudt(u,p[1:n_sde.len])
diffusion_(u,p,t) = diffusion_dudt(u,p[(n_sde.len+1):end])

display(n_sde.p)

function predict_n_sde(p)
  Array(n_sde(u0,p))
end

function loss_n_sde(p;n=100)
  samples = [predict_n_sde(p) for i in 1:n]
  means = reshape(mean.([[samples[i][j] for i in 1:length(samples)] for j in 1:length(samples[1])]),size(samples[1])...)
  vars = reshape(var.([[samples[i][j] for i in 1:length(samples)] for j in 1:length(samples[1])]),size(samples[1])...)
  loss = sum(abs2,target_data[:, 1:end] - means) + sum(abs2,sde_data_vars[:, 1:end] - vars)
  loss,means,vars
end
loss_n_sde(n_sde.p)

opt = ADAM()

cb = function (p,loss,means,vars) #callback function to observe training
  display(loss)

  pl = scatter(t[2:end],target_data[:,2:end]',label="data",markercolor=:red)
  scatter!(pl,t[2:end],means[1,:],ribbon = vars[1,:], label="pred_flux",markercolor=:blue)
  display(plot(pl, size=(1000, 600)))
  return false
end
cb(n_sde.p,loss_n_sde(n_sde.p)...)

res1 = DiffEqFlux.sciml_train((p)->loss_n_sde(p,n=10),  n_sde.p, opt, cb = cb, maxiters = 50)

