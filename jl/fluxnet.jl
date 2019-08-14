using Flux, DataFrames, CSV, Statistics

function FluxLoader()
    df = CSV.read("../demos/data/training_set.csv")
    meta = CSV.read("../demos/data/training_set_metadata.csv")
    # curves = groupby(df[: , [:object_id, :mjd, :flux]], :object_id)  # only using 3 columns
    groups = groupby(df, :object_id)
    curves = data_from_subdfs(groups)

    targets = meta[: , :target]
    classes = sort(unique(targets))
    labels = Flux.onehotbatch(targets, classes)  # Y

    label_vec = onehot_batch_to_vec(labels)

    data = zip(curves, label_vec)
    return data
end


# this is probably super inefficient and could be done w a single 'by' call
function data_from_subdfs(grouped_df)
    curves = []
    for subdf in grouped_df
        sdf_data = convert(Matrix, select(subdf, [:mjd, :flux]))

        # flatten
        sdf_data = reshape(sdf_data, length(sdf_data), 1)
        # append to lc list
        append!(curves, [sdf_data])
    end
    return curves
end


function onehot_batch_to_vec(one_hotted_batch)
    len = length(one_hotted_batch[1, :])
    label_vec = []
    for i in 1:len
        append!(label_vec, [one_hotted_batch[:, i]])
    end
    return label_vec
end



# function train()
data = FluxLoader()
model = Chain(Dense(704, 352, tanh), Dense(352, 176, tanh), Dense(176, 14), softmax)
loss(x, y) = Flux.crossentropy(model(x), y)
params = Flux.params(model)
evalcb = () -> @show(loss(X, Y))
accuracy(x, y) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))
# callback= # todo
opt = ADAM()
Flux.train!(loss, params, data, opt, cb=Flux.throttle(evalcb, 10))

accuracy(X, Y)
