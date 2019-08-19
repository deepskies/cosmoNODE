

module AutoNet
    function simple_construct(input_dim, output_dim)
        # create a list of tuples

        if output_dim == 1
            model_type = "reg"
        else
            model_type = "classify"

        
