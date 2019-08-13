using Flux, DataFrames, CSV

function FluxLoader()
    df = CSV.read("../demos/data/training_set.csv")
    meta = CSV.read("../demos/data/training_set_metadata.csv")
    curves = groupby(df, :object_id)
    targets = meta[: , :target]
    classes = sort(unique(targets))
    labels = Flux.onehotbatch(targets, classes)
    data = zip(curves, labels)
    return data
end


# function train()
data = FluxLoader()
model = Chain(Dense(704, 352, tanh), Dense(352, 176, tanh), Dense(176, 14), softmax)
loss(x, y) = Flux.crossentropy(model(x), y)
params = Flux.params(model)
evalcb = () -> @show(loss(X, Y))
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
# callback= # todo
opt = ADAM()
Flux.train!(loss, params, data, opt, cb=throttle(evalcb, 10))

accuracy(X, Y)
