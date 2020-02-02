using Flux, Plots, Statistics
using BSON: @save, @load
using CuArrays
include("utils.jl")
using .Utils


# @load "fluxnet.bson" model
data, labels = Utils.FluxLoader()
deletecols!(data, :object_id)
data  = Matrix(data)
labels = labels |> gpu 
data = data |> gpu 
dim = length(data[1, :])
out_dim = length(labels[1, :])

ex_x, ex_y = Utils.example(data, labels)

model = Chain(
    Dense(dim, 100, relu), 
    Dense(100, 100, relu), 
    Dense(100, 100, relu), 
    Dense(100, 100, relu), 
    Dense(100, out_dim), 
    softmax
    ) |> gpu
    
params = Flux.params(model)

loss(x, y) = Flux.crossentropy(model(x), y)
accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
opt = ADAM()

losses = []
accs = []

cb = function ()
    test_x, test_y = Utils.example(data, labels)
    y_hat = model(test_x)
    
    println(string("yhat:", y_hat))
    println(string("test_y", test_y))
    
    test_loss = loss(test_x, test_y)
    test_acc = accuracy(test_x, test_y)
    println(test_loss)
    
    append!(losses, test_loss)
    append!(accs, test_acc)
end

train_data = Iterators.repeated(Utils.example(data, labels), 10000)
Flux.train!(loss, params, train_data, opt, cb = cb)

accuracy(example(data, labels)...)

function plotit(arr)
    plot(scatter(arr, markersize=0.5, show=true, size=(900, 900)))
    # scatter(losses, show=true)
    # savefig("losses.png")
end

plotit(losses)
plotit(accs)

@save model "mlp_classifier.bson"