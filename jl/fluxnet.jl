module FL
    using Flux, DataFrames, CSV, Statistics
    function FluxLoader()
        df = CSV.read("../demos/data/training_set.csv")
        meta = CSV.read("../demos/data/training_set_metadata.csv")
        # curves = groupby(df[: , [:object_id, :mjd, :flux]], :object_id)  # only using 3 columns
        max_seq_len = seq_max_len_by(df)

        groups = groupby(df, :object_id)
        # includes padding
        curves = data_from_subdfs(groups, max_seq_len)

        targets = meta[: , :target]
        classes = sort(unique(targets))
        labels = Flux.onehotbatch(targets, classes)  # Y

        # padded_lcs = []
        # for curve in curves
        #     padded_lc = pad_lc(curve)
        #     append!(padded_lcs, padded_lc)
        # end

        label_vec = onehot_batch_to_vec(labels)

        # data = zip(curves, label_vec)

        return curves, label_vec
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
        return vec_lc
    end


    function LearnLC()
        data = FluxLoader()
        X, Y = data
        data_size = length(X)
        # X = convert(Array{Float32, data_size}, X)
        # Y = convert(Array{Float32, data_size}, Y)
        model = Chain(Dense(704, 352, tanh), Dense(352, 176, tanh), Dense(176, 14), softmax)
        loss(x, y) = Flux.crossentropy(model(x), y)
        params = Flux.params(model)
        evalcb = () -> @show(loss(X, Y))
        accuracy(x, y) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))
        # callback= # todo
        opt = ADAM()
        Flux.train!(loss, params, data, opt, cb=Flux.throttle(evalcb, 10))

        accuracy(X, Y)
    end
end
