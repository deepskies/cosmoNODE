module Utils 
using CSV, DataFrames, StatsBase, Flux, CuArrays

# CLASSES = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95] # N = 14
# CLASSMAP = Dict(i => j for (i, j) in zip(1:14, CLASSES))
# display(CLASSMAP)
# out_dim = length(CLASSES)

function FluxLoader(;norm::Bool=false, v::Bool=true) #;hot::Bool=true)
    df, meta = read_data()
    # dropmissing!(df)
    data = join(df, meta, on=:object_id) # has target 

    # dropping most common 
    data = data[data.target .!= 90, :]
    obj_ids = data.object_id
    
    display(describe(data))
    bands = Flux.onehotbatch(data.passband, 0:5)'
    data = hcat(data, DataFrame(bands), makeunique=true)
    
    classes = sort(unique(data.target))
    labels = Flux.onehotbatch(data.target, classes)'
    if norm
        deletecols!(data, [:passband, :target, :distmod, :object_id])
        m = Matrix(data)
        dt = fit(UnitRangeTransform, m)
        ret = DataFrame(StatsBase.transform(dt, m))
        ret.object_id = obj_ids
    else
        deletecols!(data, [:passband, :target, :distmod])
        ret = Matrix(data)
    if v
        display(size(ret))
        display(size(labels))
    end
    return ret, labels
end

function get_curve(curves)
    idx = rand(1:length(curves))
    curve = curves[idx]
    subset = view_subset(curve)

    m = Matrix(subset)'
    dt = fit(UnitRangeTransform, m)
    m = StatsBase.transform(dt, m)
    t = copy(m[1, :]) #|> gpu
    display(t)

    data = copy(m[2:end, :])
    target_data = data # |> gpu
    target_data = target_data[:, :]
    # augmented
    #   target_data = vcat(target_data,zeros(1,size(target_data,2)))
    #   target_data = vcat(target_data,zeros(1,size(target_data,2)))

    u0 = target_data[:, 1] #|> gpu
    println(string("initial conditions", u0))
    return t, target_data, u0
end

function example(data, labels)
    idx = rand(1:size(data)[1])
    Array(data[idx, :]) |> gpu, Array(labels[idx, :]) |> gpu
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