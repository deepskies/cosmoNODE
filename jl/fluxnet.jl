module FL
    using Flux, DataFrames, CSV, Statistics
    function FluxLoader()
        df = CSV.read("../demos/data/training_set.csv")

        labels = get_labels()
        curves = get_curves(df)

        # data = zip(curves, labels)
        return (curves, labels)
    end

    function get_df()
        df = CSV.read("../demos/data/training_set.csv")
        return df
    end

    function get_groups(df)
        groups = groupby(df, :object_id)
        return groups
    end

    function get_curves(df)
        groups = get_groups(df)
        max_seq_len = seq_max_len_by(df)
        curves = data_from_subdfs(groups, max_seq_len)
        # curves = hcat(Array{Float32, 1}.(reshape.(curves, :))...)
        curves = hcat(float.(reshape.(curves, :))...)
        return curves
    end

    function get_labels()
        meta = CSV.read("../demos/data/training_set_metadata.csv")
        targets = meta[: , :target]
        classes = sort(unique(targets))
        labels = Flux.onehotbatch(targets, classes)  # Y
        return labels
    end


    # TODO test if this is faster than a by(df, :object_id, nrow) call
    function seq_max_len_loop(grouped_df)
        max_seq_len = 0
        for g in grouped_df
            g_len = size(g)[1]
            # println(g_len)
            if g_len > max_seq_len
                max_seq_len = g_len
            end
        end
        return g_len
    end


    function seq_max_len_by(df)
        val_counts = by(df, :object_id, nrow)
        max_seq_len = maximum(val_counts[:, 2])
        return max_seq_len
    end

    # function gd_to_data(grouped_df)

    # this is probably super inefficient and could be done w a single 'by' call
    function data_from_subdfs(grouped_df, max_seq_len)
        curves = []
        for subdf in grouped_df
            cols = [:mjd, :flux]
            num_cols = length(cols)

            sdf_data = convert(Matrix, select(subdf, cols))

            # flatten
            sdf_data = reshape(sdf_data, length(sdf_data))
            padded_sdf = pad_lc(sdf_data, max_seq_len, num_cols)
            # append to lc list
            append!(curves, [padded_sdf])
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


    # given (n, ) array, converts to vector of max_seq_len
    function pad_lc(lc, max_seq_len, num_cols)
        lc_len = length(lc)
        vec_lc = vec(lc)
        padding = zeros((max_seq_len * num_cols) - lc_len)
        append!(vec_lc, padding)
        arr_lc = convert(Array, vec_lc)
        return arr_lc
    end


    function LearnLC()
        X, Y = FluxLoader()
        dataset = Base.Iterators.repeated((X, Y), 50)

        model = Chain(Dense(704, 352), Dense(352, 176, relu), Dense(176, 14, relu), softmax)
        loss(x, y) = Flux.mse(model(x), y)
        params = Flux.params(model)
        evalcb = () -> @show(loss(X, Y))
        accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
        opt = ADAM()

        println("abt to train")
        Flux.train!(loss, params, dataset, opt, cb=Flux.throttle(evalcb, 10))
        accuracy(X, Y)
    end
end
