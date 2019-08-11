using Flux, DataFrames, CSV

function FluxLoader()
    df = CSV.read("../demos/data/training_set.csv")
    meta = CSV.read("../demos/data/training_set_metadata.csv")
    classes = sort(unique(targets))

    labels = Flux.onehotbatch(targets, classes)
    data = zip(curves, labels)
    return data


# function train()
