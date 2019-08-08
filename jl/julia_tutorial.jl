# file for loading in data

using CSV, DataFrames, Plots

df = CSV.read('../demos/data/training_set.csv')

# these are Arrays/Vectors
all_fluxes =  df[:, :flux]
all_mjds = df[:, :mjd]

# this is a DataFrame
df_fluxes = df[:, [:flux]]

# note, vectors are 1 indexed
vector = [1, 2, 3]
# vector[0] gives error
vector[1] == 1

# look how easy this is!
plot(all_mjds[1:100], all_fluxes[1:100])
scatter(all_mjds[1:100], all_fluxes[1:100])

curves = groupby(df, :object_id)
obj = curves[1]

lc = convert(matrix, obj[:, [:mjd, :flux]])'  # ' is transpose

mjds = lc[1, :]
fluxes = lc[2, :]



# https://github.com/FluxML/model-zoo for more
using Flux, Flux.Tracker
