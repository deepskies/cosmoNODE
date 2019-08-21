# a julia module for creating Flux layers automatically
module Shape
    using Flux
    function log_dims(input_dim, output_dim, factor=2, verbose=false)
        # '''
        # mnist mlp w factor 2
        # [784, 397, 203, 106, 58, 34, 22, 16, 13, 11, 10]
        # '''
        dims = []
        dim = input_dim
        delta = input_dim - output_dim

        while dim > output_dim
            append!(dims, dim)
            dim = Int.(floor(delta // factor) + output_dim)
            delta = dim - output_dim
        end
        append!(dims, output_dim)
        if verbose
            print(dims)
        end
        return dims
    end

    function get_layers(layer_dims)
        # assuming dense layers rn
        layers = []
        num_layers = length(layer_dims)
        for i in 1:num_layers
            if i == num_layers
                break
            end
            # dim_1 = layer_dims[i]
            # dim_2 = layer_dims[i + 1]
            append!(layers, [Dense(layer_dims[i], layer_dims[i + 1])])
        end
        return layers
    end
end
