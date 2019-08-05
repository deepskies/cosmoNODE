# file for loading in data

using CSV, DataFrames, Plots

df = CSV.read('../demos/data/training_set.csv')

# these are Arrays
fluxes =  df[:, :flux]
mjds = df[:, :mjd]

# this is a DataFrame
df_fluxes = df[:, [:flux]]

# note, vectors are 1 indexed
vector = [1, 2, 3]
# vector[0] gives error
vector[1] == 1

# look how easy this is!
plot(mjds[1:100], fluxes[1:100])
scatter(mjds[1:100], fluxes[1:100])


# https://github.com/FluxML/model-zoo for more
using Flux, Flux.Tracker
