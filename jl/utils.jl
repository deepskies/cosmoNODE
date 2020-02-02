module Utils 
using CSV, DataFrames, StatsBase, Flux, CuArrays

# CLASSES = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95] # N = 14
# CLASSMAP = Dict(i => j for (i, j) in zip(1:14, CLASSES))
# display(CLASSMAP)
# out_dim = length(CLASSES)

function FluxLoader(;norm::Bool=false, v::Bool=true) #;hot::Bool=true)
    df, meta = read_data()
    # data = join(df, meta, on=:object_id) # has target 
    # df = dropmissing(data)
    data = dropmissing(df)

    # dropping most common 
    # data = data[data.target .!= 90, :]
    # obj_ids = data.object_id
    
    bands = Flux.onehotbatch(data.passband, 0:5)'
    data = hcat(data, DataFrame(bands), makeunique=true)
    
    classes = sort(unique(meta.target))
    labels = Flux.onehotbatch(meta.target, classes)'
    deletecols!(data, [:passband])
    # deletecols!(data, [:passband, :target, :distmod])
    if norm
        deletecols!(data, [:object_id])
        data = normdf(data)
        data.object_id = obj_ids
    end
    if v
        display(size(data))
        display(size(labels))
    end
    return data, labels
end

function get_curve(curves::GroupedDataFrame, idx::Union{Int, DataFrames.GroupKey}; norm::Bool=false)::Matrix
    # like pytorch dataset __getitem__
    curve = DataFrame(curves[idx])
    deletecols!(curve, :object_id)
    subset = copy(view_subset(curve))
    subset = normdf(subset)
    # subset = normdf(curve)
    Matrix(subset)
end


function example(data, labels)
    # __getitem__ for rows 
    idx = rand(1:size(data)[1])
    Array(data[idx, :]), Array(labels[idx, :])
end 


function normdf(df::DataFrame)::DataFrame
    cols = names(df)
    m = Matrix(df)'
    dt = fit(UnitRangeTransform, m)
    DataFrame(StatsBase.transform(dt, m)) #, names=cols)
end

function read_data()
    path = "/home/sippycups/D/kaggle/PLAsTiCC-2018/"
    df = CSV.read(string(path, "training_set.csv"))
    meta = CSV.read(string(path, "training_set_metadata.csv"))
    return df, meta
end

function view_subset(df, start_frac=0.75, end_frac=1)
  len = size(df)[1]
  view_start = Int(round(len * start_frac))
  view_end = Int(round(len * end_frac))
  stride_amt = 2
  subset = view(df, view_start:stride_amt:view_end, :)
  return subset
end
# ------

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


# given (n, 1) array, converts to vector of max_seq_len
function pad_lc(lc, max_seq_len)
    veclc = vec(lc)
end




end