# file for loading in data

using CSV, DataFrames, Plots

df = CSV.read("../demos/data/training_set.csv")
meta = CSV.read("../demos/data/training_set_metadata.csv")
# these are Arrays/Vectors
all_fluxes =  df[:, :flux]
all_mjds = df[:, :mjd]

# this is a DataFrame
df_fluxes = df[:, [:flux]]

# note, vectors are 1 indexed
vector = [1, 2, 3]
# vector[0] gives error
vector[1] == 1  # true

# look how easy this is!
plot(all_mjds[1:100], all_fluxes[1:100])
scatter(all_mjds[1:100], all_fluxes[1:100])

curves = groupby(df, :object_id)
targets = meta[: , :target]

classes = sort(unique(targets))

labels = Flux.onehotbatch(targets, classes)
data = zip(curves, labels)


curve = curves[1]

lc = convert(Matrix, obj[:, [:mjd, :flux]])'  # ' is transpose
flat_lc = vec(lc)

mjds = lc[1, :]  #
fluxes = lc[2, :]


# todo pad sequence
# todo one hot


# https://github.com/FluxML/model-zoo for more
using Flux, Flux.Tracker

function train(data):

    for d in data
        # d looks like (data, labels)
        println(d)
        l = Flux.crossentropy(d)
