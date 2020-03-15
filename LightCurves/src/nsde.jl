using CSV, DataFrames
using DifferentialEquations, Flux
using DiffEqFlux
using DiffEqFlux: FastChain, FastDense
using Plots, BenchmarkTools
using StochasticDiffEq
using DiffEqBase.EnsembleAnalysis
using StatsBase, Statistics
include("utils.jl")
using .Utils

function read_data()
    path = "/home/sippycups/D/kaggle/PLAsTiCC-2018/"
    df = CSV.read(string(path, "training_set.csv"))
    meta = CSV.read(string(path, "training_set_metadata.csv"))
    return df, meta
end

function view_subset(df, start_frac = 0.50, end_frac = 1, stride_amt=2)
	len = size(df)[1]
	view_start = Int(round(len * start_frac))
	view_end = Int(round(len * end_frac))
	subset = view(df, view_start:stride_amt:view_end, :)
	return subset
end

df, meta = read_data()
curve_data = groupby(df, [:object_id, :passband])
idx = rand(1:length(curve_data))
m = copy(view_subset(curve_data[2197]))
# m = copy(curve_data[2197])

err = copy(m.flux_err)
t = copy((m.mjd .- m.mjd[1]) ./ 1e3)
tspan = (t[1], t[end])

target_data = Matrix(select!(m, :flux))'
u0 = target_data[:, 1]
dim = length(u0)

sde_data_vars = zeros(dim, length(t))

drift_dudt = FastChain(
	FastDense(dim, 16, tanh),
	# FastDense(16, 16, tanh),
	FastDense(16, dim)
) #|> gpu

diffusion_dudt = FastChain(FastDense(dim,dim))

n_sde = NeuralDSDE(drift_dudt,diffusion_dudt,tspan,SOSRI(),saveat=t,reltol=1e-2,abstol=err)
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
  loss = sum(abs2,target_data - means) + sum(abs2,sde_data_vars - vars)
  loss,means,vars
end

loss_n_sde(n_sde.p)

opt = ADAM()

cb = function (p,loss,means,vars) #callback function to observe training
  display(loss)

  pl = scatter(t,target_data',label="data",markercolor=:red)
  scatter!(pl,t,means[1,:],ribbon = vars[1,:], label="pred_flux",markercolor=:blue)
  display(plot(pl, size=(1000, 600)))
  return false
end
cb(n_sde.p,loss_n_sde(n_sde.p)...)

res1 = DiffEqFlux.sciml_train((p)->loss_n_sde(p,n=30),  n_sde.p, opt, cb = cb, maxiters = 50)

