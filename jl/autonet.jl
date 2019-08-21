module Auto
    using Flux

    include("./Shape.jl")
    using .Shape

    function mlp(input_dim=784, output_dim=10, factor=2)
        # if output_dim == 1
        #     model_type = "reg"
        # else
        #     model_type = "classify"
        # end

        dims = Shape.log_dims(input_dim, output_dim)
        layers = Shape.get_layers(dims)
        model(x) = foldl((x, m) -> m(x), layers, init = x)
        return model
    end
end
