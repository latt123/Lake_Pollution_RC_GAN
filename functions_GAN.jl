# # Importing Packages
using Flux
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy, logitcrossentropy
using Statistics
using Parameters: @with_kw
using Zygote
using Distributions, Random
using StatsPlots
using KernelDensity
using DataFrames, DataFramesMeta, CSV, Dates
using BSON: @save, @load



# # Functions to prepare dataframe for training
"""
## Sequence of dates such that:
    Start Date: Earliest date in df
    End Date : Latest date in df
    Interval : Δt
"""
function date_Full(df, Δt)
    return df.Date[1]:Δt:df.Date[end]
end


"""
## 'Fill out' a data set to include all dates in date_full. Then Impute.interp to 'smooth' missing values.
"""
function interp_df(df, date_full)
    ## Change any colomns with type Integer to Float 64 (to be able to use Interp function)
    df_Float32 = hcat(DataFrame(Date=df.Date), Float32.(df[:,2:end]))
    
    ## New data frame which includes the missing dates too
    df_date_full = leftjoin(DataFrame(Date=date_full), df_Float32,on=:Date)

    ## Sort by date (missing dates added to the end by left join.)
    df_date_full_sorted = sort(df_date_full, :Date)

    ## Apply Impute.interp_df
    df_impute = Impute.interp(df_date_full_sorted)

    return df_impute
end



"""
Create new dataframe of length 'percent' of input 'df'. 
First date in new data frame is the first value after or equal  to 'start_date' in dates of input 'df'.
"""
function df_section(df, percent, start_date)
    index_start = range(1,size(df,1))[df.Date .>= start_date][1]
    index_end = index_start + Int(floor(percent*size(df,1))) - 1
    dataframe = df[index_start:index_end,:]
    return dataframe
end

"""
Choise of kernels: 
    - uniform,
    - triangular,
    - gaussian,
    - epanechnikov,
    - biweight,
    - triweight,
    - tricube,
    - cosine,
    - logistic
"""
function df_smoothing(df, kernel, h)
    SK = SmoothingKernel(kernel, h.value)  
    dataframe_smooth = copy(df)
    x = DateTime.(df.Date)
    if size(df,2) > 1
        for i in 1:(size(df,2)-1)
            y = df[:,i+1]*1f0
            dataframe_smooth[:,i+1] = mean(y, datetime2unix.(x)*1f0, SK)
        end
    end
    return dataframe_smooth
end

function pred_cond_df(pred_features, cond_features, whole_df)
    prop_names = propertynames(whole_df)
    df_cond_feat = DataFrame(Date = whole_df.Date)
    df_pred_feat = DataFrame(Date = whole_df.Date)
    for i in 1:size(cond_features,1)
        index_cond = range(1,size(prop_names,1))[prop_names .== cond_features[i]][1]
        df_cond_feat[!, cond_features[i]] = Vector{Float32}(whole_df[:,index_cond])
    end
    for i in 1:size(pred_features,1)
        index_pred = range(1,size(prop_names,1))[prop_names .== pred_features[i]][1]
        df_pred_feat[!, pred_features[i]] = Vector{Float32}(whole_df[:,index_pred])
    end
    return df_pred_feat, df_cond_feat
end

function normalise_df(df_original)
    df_norm = copy(df_original)
    features = size(df_original, 2) - 1
    for j in 1:features
        v = df_norm[:, j+1]
        v = v .- min(v...)
        v = v./ max(v...)
        df_norm[:, j+1] = v
    end
    return df_norm
end

function denormalise(df_original, df_norm)
    df_denorm = copy(df_norm)
    features = size(df_original, 2) - 1
    for j in 1:features
        v = df_original[:, j+1]
        min_v = min(v...)
        v = v .- min_v
        max_v = max(v...)
        df_denorm[:, j+1] = df_norm[:, j+1].*max_v .+ min_v
    end
    return df_denorm
end

# # Creating Generator 

"""
With elements:

- g_cond - function to permutate the conditional input, a sequence of the previous data, into the same dimension as noise input
- g_common - function that takes as inputs the permuted conditional and noise inputs combined and outputs a prediction for the following value of the sequence

The parameters of these two functions will be optimised when the Generator is trained.
"""
struct Generator
    g_cond
    g_common       
end


"""
Inputs:
- seq_len - the length of the conditional input, a sequence of previous data
- input_features - the number of features to conditon by, both to be predicted and just for conditioning
- feautres_pred - the number of features to be generated
- noise_dim - the dimension of the noise input
- rnn_units - the units of the GRU layers in the g_cond function
- dense_units - the units of the dense layer in the g_common function
Outputs:
- Generator structure with elements g_cond, g_common

For g_cond:
- Input Dim (input_features_cond,batch_size,seq_len)
- Output Dim (rnn_units[end], batch_size)

For g_common:
- Input Dim (rnn_units[end] + noise_dim, batch_size)
- Output Dim (1, feautres_pred, batch_size)
"""
function generator(seq_len, input_features, features_pred, noise_dim, rnn_layer, rnn_units, dense_units)

    if rnn_layer == "GRU"
        l1 = GRU(input_features, rnn_units[1])
        for i in 1:(size(rnn_units, 1)-1)
            l = GRU(rnn_units[i], rnn_units[i+1])
            l1 = Flux.Chain(l1, l)
        end
        g_cond = Flux.Chain(l1, x->permutedims(x, (3, 2, 1)), Dense(seq_len=>1), x -> dropdims(x; dims=1),x->permutedims(x, (2, 1)))
    else
        if rnn_layer == "LSTM"
            l1 = LSTM(input_features, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = LSTM(rnn_units[i], rnn_units[i+1])
                l1 = Flux.Chain(l1, l)
            end
            g_cond = Flux.Chain(l1, x->permutedims(x, (3, 2, 1)), Dense(seq_len=>1), x -> dropdims(x; dims=1),x->permutedims(x, (2, 1)))
        else 
            l1 = RNN(input_features, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = RNN(rnn_units[i], rnn_units[i+1])
                l1 = Flux.Chain(l1, l)
            end
            g_cond = Flux.Chain(l1, x->permutedims(x, (3, 2, 1)), Dense(seq_len=>1), x -> dropdims(x; dims=1),x->permutedims(x, (2, 1)))
        end
    end
    
    g_common = Flux.Chain(Dense(noise_dim+rnn_units[end] => dense_units), Dense(dense_units => features_pred), x->reshape(x, size(x, 1), size(x, 2) ,1))

    Generator(g_cond, g_common)
end

"""
## Using just the last value of the output of RNN layers - assuming this represents the values for 't+1' after the input sequence ending for time 't'

Inputs:
- seq_len - the length of the conditional input, a sequence of previous data
- input_features - the number of features to conditon by, both to be predicted and just for conditioning
- feautres_pred - the number of features to be generated
- noise_dim - the dimension of the noise input
- rnn_units - the units of the GRU layers in the g_cond function
- dense_units - the units of the dense layer in the g_common function
Outputs:
- Generator structure with elements g_cond, g_common

For g_cond:
- Input Dim (input_features ,batch_size,seq_len)
- Output Dim (rnn_units[end], batch_size)

For g_common:
- Input Dim (rnn_units[end] + noise_dim, batch_size)
- Output Dim (1, feautres_pred, batch_size)
"""
function generator_opt2(input_features, features_pred, noise_dim, rnn_layer,rnn_units, dense_units)
    if rnn_layer == "GRU"
        g_cond = GRU(input_features, rnn_units[1])
        for i in 1:(size(rnn_units, 1)-1)
            l = GRU(rnn_units[i], rnn_units[i+1])
            g_cond  = Flux.Chain(g_cond, l)
        end
    else
        if rnn_layer == "LSTM"
            g_cond = LSTM(input_features, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = LSTM(rnn_units[i], rnn_units[i+1])
                g_cond  = Flux.Chain(g_cond, l)
            end
        else
            g_cond = RNN(input_features, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = RNN(rnn_units[i], rnn_units[i+1])
                g_cond  = Flux.Chain(g_cond, l)
            end
        end
    end
    g_cond = Chain(g_cond,x->x[:,:,end])
    g_common = Flux.Chain(Dense(noise_dim+rnn_units[end] => dense_units), Dense(dense_units => features_pred), x->reshape(x, size(x, 1), size(x, 2) ,1))
    Generator(g_cond, g_common)
end


"""
Inputs:
- input_cond_feat - a sequence of the previous data that will condtion the model by but NOT predict Dimension (seq_len, features_cond, batch_size)
- input_cond_feat - a sequence of the previous data that will condtion the model by AND predict. Dimension (seq_len, features_pred, batch_size)
- input_noise -  Dimension: (noise_dim, batch_size)
Return: 
- Prediction for following value in the sequence, Dimension (1, features, n)
"""
function (m::Generator)(input_cond_feat, input_pred_feat, input_noise)
    ## First transfrom the conditional data into common shape
    t = cat(m.g_cond(cat(input_cond_feat, input_pred_feat, dims=1)), input_noise, dims=1)
    ## Then apply common network on the transformed conditional and prediction to output prob the data is real. 
    return m.g_common(t)
end


# # Creating Discriminator
"""
With element:
- d_common - function that takes as inputs the permuted conditional and noise inputs combined and outputs a prediction for the following value of the sequence

The parameters of this function will be optimised when the Discriminator is trained.
"""
struct Discriminator
    d_cond_feat
    d_pred_feat
    d_common
end


"""
Inputs:
- seq_len - the length of the conditional input, a sequence of previous data
- features_cond - the number of features to conditon by
- feautres_pred - the number of features to be generated
- dense_units - the units of the dense layer in the d_common function
Outputs:
- Discriminator structure with element d_common
"""
function discriminator(seq_len, features_cond, features_pred, rnn_layer, rnn_units, dense_units)
    if rnn_layer == "GRU"
        l_pred = GRU(features_pred, rnn_units[1])
        for i in 1:(size(rnn_units, 1)-1)
            l = GRU(rnn_units[i], rnn_units[i+1])
            l_pred = Flux.Chain(l_pred, l)
        end
        d_pred_feat = Flux.Chain(l_pred, x->permutedims(x, (3, 2, 1)), Dense(seq_len+1=>dense_units),Dense(dense_units => 1), x -> dropdims(x; dims=1),x->permutedims(x, (2, 1)))
        ## Now dim is (rnn_units[end], batch_size)

        l_cond = GRU(features_cond, rnn_units[1])
        for i in 1:(size(rnn_units, 1)-1)
            l = GRU(rnn_units[i], rnn_units[i+1])
            l_cond = Flux.Chain(l_cond, l)
        end
    else
        if rnn_layer == "LSTM"
            l_pred = LSTM(features_pred, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = LSTM(rnn_units[i], rnn_units[i+1])
                l_pred = Flux.Chain(l_pred, l)
            end
            d_pred_feat = Flux.Chain(l_pred, x->permutedims(x, (3, 2, 1)), Dense(seq_len+1=>dense_units),Dense(dense_units => 1), x -> dropdims(x; dims=1),x->permutedims(x, (2, 1)))
            ## Now dim is (rnn_units[end], batch_size)
    
            l_cond = LSTM(features_cond, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = LSTM(rnn_units[i], rnn_units[i+1])
                l_cond = Flux.Chain(l_cond, l)
            end
        else
            l_pred = RNN(features_pred, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = RNN(rnn_units[i], rnn_units[i+1])
                l_pred = Flux.Chain(l_pred, l)
            end
            d_pred_feat = Flux.Chain(l_pred, x->permutedims(x, (3, 2, 1)), Dense(seq_len+1=>dense_units),Dense(dense_units => 1), x -> dropdims(x; dims=1),x->permutedims(x, (2, 1)))
            ## Now dim is (rnn_units[end], batch_size)
    
            l_cond = RNN(features_cond, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = RNN(rnn_units[i], rnn_units[i+1])
                l_cond = Flux.Chain(l_cond, l)
            end
        end
    end

    if features_cond == 0
        d_common =  Flux.Chain(Dense(rnn_units[end]=> dense_units), Dense(dense_units=>1), sigmoid)
        d_cond_feat = Flux.Chain(x->x)
    else
        d_cond_feat = Flux.Chain(l_cond, x->permutedims(x, (3, 2, 1)), Dense(seq_len=>dense_units),Dense(dense_units => 1), x -> dropdims(x; dims=1),x->permutedims(x, (2, 1)))  ## Now dim is (rnn_units[end], batch_size)
        d_common = Flux.Chain(Dense(rnn_units[end]*2=> dense_units), Dense(dense_units=>1), sigmoid)
    end    
    Discriminator(d_cond_feat, d_pred_feat, d_common)
end

"""
Inputs:
- input_cond_feat - a sequence of the previous data that will condtion the model by but NOT predict Dimension (seq_len, features_cond, batch_size)
- input_cond_feat - a sequence of the previous data that will condtion the model by AND predict. Dimension (seq_len, features_pred, batch_size)
- post_pred_feat -  a proposed or real value that follows the sequence for the predicted features. Of dimension (1, features, n)
Return: 
- The probability that the sequence followed by the 'post' value is from the real data set
"""
function (m::Discriminator)(input_cond_feat, input_pred_feat, post_pred_feat)
    seq_complete_pred_feat = cat(input_pred_feat, post_pred_feat, dims=3)
    if size(input_cond_feat,1) == 0
        t = m.d_pred_feat(seq_complete_pred_feat)
    else
        t = cat(m.d_cond_feat(input_cond_feat), m.d_pred_feat(seq_complete_pred_feat), dims=1)
    end
    return m.d_common(t)
end



"""
## Using just the last value of the output of RNN layers - assuming this represents the values for 't+1' after the input sequence ending for time 't'

Inputs:
- seq_len - the length of the conditional input, a sequence of previous data
- features_cond - the number of features to conditon by
- feautres_pred - the number of features to be generated
- dense_units - the units of the dense layer in the d_common function
Outputs:
- Discriminator structure with element d_common
"""
function discriminator_opt2(features_cond, features_pred, rnn_layer, rnn_units, dense_units)

    if rnn_layer == "GRU"
        d_pred_feat = GRU(features_pred, rnn_units[1])
        for i in 1:(size(rnn_units, 1)-1)
            l = GRU(rnn_units[i], rnn_units[i+1])
            d_pred_feat  = Flux.Chain(d_pred_feat, l)
        end  
        d_pred_feat = Chain(d_pred_feat, x->x[:,:,end]) 
        if features_cond == 0
            d_common =  Flux.Chain(Dense(rnn_units[end]=> dense_units), Dense(dense_units=>1), sigmoid)
            d_cond_feat = Flux.Chain(x->x)
        else
            d_cond_feat = GRU(features_cond, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = GRU(rnn_units[i], rnn_units[i+1])
                d_cond_feat  = Flux.Chain(d_cond_feat, l)
            end  
            d_cond_feat = Chain(d_cond_feat, x->x[:,:,end]) 
            d_common = Flux.Chain(Dense(rnn_units[end]*2=> dense_units), Dense(dense_units=>1), sigmoid)
        end    
    else
        if rnn_layer == "LSTM"
            d_pred_feat = LSTM(features_pred, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = LSTM(rnn_units[i], rnn_units[i+1])
                d_pred_feat  = Flux.Chain(d_pred_feat, l)
            end  
            d_pred_feat = Chain(d_pred_feat, x->x[:,:,end]) 
            if features_cond == 0
                d_common =  Flux.Chain(Dense(rnn_units[end]=> dense_units), Dense(dense_units=>1), sigmoid)
                d_cond_feat = Flux.Chain(x->x)
            else
                d_cond_feat = LSTM(features_cond, rnn_units[1])
                for i in 1:(size(rnn_units, 1)-1)
                    l = LSTM(rnn_units[i], rnn_units[i+1])
                    d_cond_feat  = Flux.Chain(d_cond_feat, l)
                end  
                d_cond_feat = Chain(d_cond_feat, x->x[:,:,end]) 
                d_common = Flux.Chain(Dense(rnn_units[end]*2=> dense_units), Dense(dense_units=>1), sigmoid)
            end    
        else
            d_pred_feat = RNN(features_pred, rnn_units[1])
            for i in 1:(size(rnn_units, 1)-1)
                l = RNN(rnn_units[i], rnn_units[i+1])
                d_pred_feat  = Flux.Chain(d_pred_feat, l)
            end  
            d_pred_feat = Chain(d_pred_feat, x->x[:,:,end]) 
            if features_cond == 0
                d_common =  Flux.Chain(Dense(rnn_units[end]=> dense_units), Dense(dense_units=>1), sigmoid)
                d_cond_feat = Flux.Chain(x->x)
            else
                d_cond_feat = RNN(features_cond, rnn_units[1])
                for i in 1:(size(rnn_units, 1)-1)
                    l = RNN(rnn_units[i], rnn_units[i+1])
                    d_cond_feat  = Flux.Chain(d_cond_feat, l)
                end  
                d_cond_feat = Chain(d_cond_feat, x->x[:,:,end]) 
                d_common = Flux.Chain(Dense(rnn_units[end]*2=> dense_units), Dense(dense_units=>1), sigmoid)
            end    
        end
    end

    d_common = Flux.Chain(Dense(rnn_units[end]+features_pred => dense_units), Dense(dense_units=>1), sigmoid)   
    Discriminator(d_cond_feat, d_pred_feat, d_common)
end

# # Loss Functions 

"""
## Generator Loss 
Input:
- y_pred - generated prediction for the following value in a sequence. Dimension (1, features, n)
- y_real - the real following value in a sequence. Dimension (1, features, n) (Array of dimension (1, features) is repeated n times)
- pred_prob - the probability that y_pred, combined with the previous sequence, comes from the real data set, as judged by the discriminator
- w_adv - the weight of the adversial loss towards the output
- w_pred - the weight of the prediction loss towards the output

Output:
Sum of:
- w_adv*adversial_loss - distance (logitbinarycrossentropy) between pred_prob and 1. The Generator wants to minimise this distance (convince the Discriminator the generator values are real)
- w_pred*prediction_loss - mean squared error between the real values and prediction
"""
function gen_loss(y_pred, y_real, pred_prob, w_adv, w_pred) 
    adversial_loss = logitbinarycrossentropy(pred_prob, 1f0) 
    prediction_loss = Flux.mse(y_pred, y_real) 
    return adversial_loss*w_adv + prediction_loss*w_pred 
end

"""
## Discriminator Loss
Input:
- pred_prob - the probability that y_pred, combined with the previous sequence, comes from the real data set, as judged by the Discriminator
- real_prob - the probability that y_real, combined with the previous sequence, comes from the real data set, as judged by the Discriminator

Output:
Sum of:
- real_loss - distance (logitbinarycrossentropy) between real_prob and 1. Reducing this means the discriminator is better at classifiying correctly.
- pred_loss - distance (logitbinarycrossentropy) between pred_prob and 0. Reducing this means the discriminator is better at classifiying correctly.
"""
function discr_loss(pred_prob, real_prob)
    real_loss = logitbinarycrossentropy(real_prob, 1f0) 
    pred_loss = logitbinarycrossentropy(pred_prob, 0f0)   
    return real_loss + pred_loss
end


"""
## Test loss
Test Loss of the Generator for a fixed 'test' set that is not used in training. 
Inputs:
- gen - Generator
- ts_test_cond_feat - the subsection of conditonal NOT PREDICTION features of a 'test set' (of data set not used for training)
- ts_test_pred_feat - the subsection of features to be predicted of a 'test set' (of data set not used for training)
- seq_len - the length of sequence of previous data from which a prediction will be generated
- samp_size - number of generated prediction to produce (together they'll be used to approximate a density)
- noise_dim - dimension of the noise input to the Generator
- noise_sd - standard deviation of the noise input to the Generator

Outputs:
For each feature, the mean of the log of pdf for each real point in the series with respect to a distrubution created by the generated using the previous value as conditional input.
"""
function test_loss(gen, test_cond_feat_array, test_pred_feat_array, seq_len, samp_size, noise_dim, noise_sd)
    len_test = size(test_pred_feat_array, 1)
    pdf_test = zeros(Float32, size(test_pred_feat_array, 2), len_test-seq_len)
    t = 1
    for i in (seq_len+1):(len_test)
        input_cond_feat = zeros(Float32, size(test_cond_feat_array, 2), samp_size, seq_len)
        input_pred_feat = zeros(Float32, size(test_pred_feat_array, 2), samp_size, seq_len)
        for j in 1:samp_size
            input_cond_feat[:, j, :] = permutedims(test_cond_feat_array[range(i-seq_len, i-1), :],(2,1))
            input_pred_feat[:, j, :] = permutedims(test_pred_feat_array[range(i-seq_len, i-1), :],(2,1))
        end
        input_noise = randn(Float32, noise_dim, samp_size)*noise_sd
        ## y is the prediction for i 
        Flux.reset!(gen.g_cond)
        y = gen(input_cond_feat, input_pred_feat, input_noise) ## output features, batchsize, seq_len (1)
        for k in 1:size(y, 1)
            U = kde(y[k, :, 1])
            pdf_test[k, t] = pdf(U, test_pred_feat_array[i, k]) +  Float32(10^(-7))
        end
        t = t+1
    end
    return -mean(log.(pdf_test), dims=2)
end


# # Functions for training

#= First for seperating the data set into batches
Identify the indexes of which dates theres enough history for that they can be used in model training (allowing for gaps in the data).
Then seperate this indexes into batches. =# 


"""
Inputs:
- df - DataFrame, expect 'date' colomn for dates
- batch_size 
- histroy_time - Date/Time format for the length of history

"""
function batch_indexes(df, batch_size, history_time)
    tbm = df.Date[2] - df.Date[1] ## time between measurements
    expected_len = Int64.(history_time/tbm) ## length of history (seq_len)
    good_indexes = []
    ## Can't  use last value
    for i in 1:(size(df, 1)-1)
        now = df.Date[i]
        history = @subset(df, now .- history_time .<=  :Date .<  now )
        len_hist = size(history, 1)
        if len_hist == expected_len
            append!(good_indexes, i)
        end
    end
    len = size(good_indexes, 1)
    num_batch = Int(ceil(len/batch_size))
    batch_index = zeros(Int, num_batch, batch_size)
    ## Each row will be the indexes of a single batch.
    for i in 1:len
        batch_index[i] = good_indexes[i]
    end
    remainder = num_batch*batch_size - len
    batch_index[range(end-remainder+1, end)] = rand(good_indexes, remainder)
    return batch_index, num_batch
end

"""
## Extract batch 'i' from the data set. 
Input: 
- indexes -  A matrix of indexes, element in row i collomn j gives the index for the row of 'data' that will be the jth element of batch i in the dataset when seperated into batches. Dimension (num_batches, batch_size)
- index_i - The row of indexes from which the batch should be made
- seq_len - the length of sequence of previous data from which a prediction will be generated
- cond_feat_array - Float32 array of only conditional features
- pred_feat_array - Float32 array of prediction features

Output:
- batch - an array of dimension (features, batch_sizeseq_len +1, ) where (:, 1, :) is the first element of the batch. The sequence of data (plus real value) to be inputed to Generator
"""
function create_batch(indexes, index_i, seq_len, cond_feat_array, pred_feat_array)
    batch_size = size(indexes, 2)
    features_cond = size(cond_feat_array, 2)
    features_pred = size(pred_feat_array, 2)
    cond_batch = zeros(Float32, features_cond, batch_size, seq_len)
    pred_batch = zeros(Float32, features_pred, batch_size, seq_len+1)

    if features_cond > 0 
        for j in 1:batch_size
            index_ts = indexes[index_i, j]
            cond_batch[:, j, :] = permutedims(cond_feat_array[range(index_ts-seq_len, index_ts-1), :],(2,1))
            pred_batch[:, j, :] = permutedims(pred_feat_array[range(index_ts-seq_len, index_ts), :],(2,1))
        end
    else
        for j in 1:batch_size
            index_ts = indexes[index_i, j]
            pred_batch[:, j, :] = permutedims(pred_feat_array[range(index_ts-seq_len, index_ts), :],(2,1))
        end
    end
    return cond_batch, pred_batch
end 


"""
## Update Discriminator parameters 
Inputs:
- input_cond - a sequence of the previous data to the value that would like to predicted. Dimension (seq_len, features, n)
- y_real - the real following value in a sequence. Dimension (1, features, n) (Array of dimension (1, features) is repeated n times)
- y_pred - generated prediction for the following value in a sequence. Dimension (1, features, n)
- opt_discr - optimiser used to update gradients
- discr - Discriminator

Output:
- loss - discriminator loss between the two outputs of the discriminator at y_pred and y_real with input_cond BEFORE the parameters are udpated
"""
function train_discr(input_cond_feat, input_pred_feat, y_real, y_pred, opt_discr, discr)
    if size(input_cond_feat,2) == 0
        ps = Flux.params(discr.d_common,discr.d_pred_feat)
    else
        ps = Flux.params(discr.d_common,discr.d_cond_feat,discr.d_pred_feat)
    end
    loss, back = Zygote.pullback(ps) do 
        discr_loss(discr(input_cond_feat, input_pred_feat, y_pred), discr(input_cond_feat, input_pred_feat, y_real))
    end
    grads = back(1f0) ## Derivative of discr_loss function evaluated at the parameters
    update!(opt_discr, ps, grads)
    return loss
end 
Zygote.@nograd train_discr

"""
## Updates the Discriminator and Generator for one 'batch' of data.
Inputs:
- data - a single batch extracted from the data set. Dimension (seq_len, features, batch_size)
- noise - to be used in Generator. Dimension (noise_dim, batch_size)
- gen - Generator
- discr - Discriminator
- opt_gen - optimiser used to update Generator gradients
- opt_discr - optimiser used to update Discriminator gradients
- w_adv - the weight of the adversial loss towards the output
- w_pred - the weight of the prediction loss towards the output

Outputs:
loss - A dictionary with the Generator, "gen" loss and Discriminator "discr" loss at this iteration of training
"""
function train_step(data_cond_feat, data_pred_feat, noise, gen, discr, opt_gen, opt_discr, w_adv, w_pred)
    
    ## Extract input sequence to be used as a condtion and the target sequence
    input_cond_feat = data_cond_feat
    input_pred_feat = data_pred_feat[:, :, range(1, end-1)]
    y_real = reshape(data_pred_feat[:, :, end], size(data_pred_feat, 1), size(data_pred_feat, 2), 1)

    ps = Flux.params(gen.g_cond, gen.g_common)
    loss = Dict()
    
    ## Reset the memory parameters of both components
    Flux.reset!(gen.g_cond)
    Flux.reset!(gen.g_common)
    Flux.reset!(discr.d_cond_feat)
    Flux.reset!(discr.d_pred_feat)
    Flux.reset!(discr.d_common)

    ## Train Generator
    ps = Flux.params(gen.g_cond, gen.g_common)
    loss["gen"], back = Zygote.pullback(ps) do
        y_pred = gen(input_cond_feat, input_pred_feat, noise)
        loss["discr"] = train_discr(input_cond_feat, input_pred_feat, y_real, y_pred, opt_discr, discr)
        gen_loss(y_pred, y_real, discr(input_cond_feat, input_pred_feat, y_pred), w_adv, w_pred)
    end
    grads = back(1f0) 
    update!(opt_gen, ps, grads)
    return loss
end




"""
## Train the Generator and Discriminator for the data set using the specificed Hyperparameters
Inputs: 
- df_cond_feat - the dataset of conditional features (NOT PREDICTION) from which the Generator and Discriminator will be trained
- df_pred_feat - the dataset of features to be predicted from which the Generator and Discriminator will be trained
- Hyperparameters - Hyperparameters for the size of the test set, optimisers, archicteture of the Generator and Discriminator, weights of generator loss function and training 

Outputs:
- loss_train_step - generator and discriminator training loss at each step of Training
- loss_train_epoch - generator and discriminator training loss after each epoch
- loss_test_epoch - test_loss after each epoch
- gen - Generator that has been trained
- discr - Discriminator that has been trained
"""
function train(model_name, df_cond_feat, df_pred_feat, hyparams, gen, discr)

    ## Extract useful parameters from Hyperparameters and dataset dimensions
    
    ## 1/Training parameters
    test_percent = hyparams.test_percent
    test_samp_size = hyparams.test_samp_size
    lr_gen = hyparams.lr_gen
    lr_discr = hyparams.lr_discr
    epochs = hyparams.epochs
    info_freq = hyparams.info_freq
    batch_size = hyparams.batch_size
    save_model_freq = hyparams.save_model_freq

    ## 2/Gen and Discr Parameters
    gen_rnn_units = hyparams.gen_rnn_units
    gen_dense_units = hyparams.gen_dense_units
    discr_rnn_units = hyparams.discr_rnn_units
    discr_dense_units = hyparams.discr_dense_units
    noise_dim = hyparams.noise_dim
    noise_sd = hyparams.noise_sd

    ## 3/ Lose functions
    w_adv = hyparams.w_adv
    w_pred = hyparams.w_pred

    ## 4/ About data
    history_time = hyparams.history_time
    seq_len = Int.(history_time/(df_pred_feat.Date[2] - df_pred_feat.Date[1]))
    features_cond = size(df_cond_feat, 2) - 1 
    features_pred = size(df_pred_feat, 2) - 1 
    input_features = features_cond + features_pred
    len = size(df_pred_feat, 1)
    train_len = Int(floor((1-test_percent)*len))

    ## Create test and train data frames
    train_df_cond_feat= df_cond_feat[range(1, train_len), :]
    test_df_cond_feat= df_cond_feat[range(train_len+1-seq_len, len), :]

    train_df_pred_feat= df_pred_feat[range(1, train_len), :]
    test_df_pred_feat= df_pred_feat[range(train_len+1-seq_len, len), :]

    ## Create test and train matrixes (no DateTime collomn)
    if size(train_df_cond_feat,2) == 1
        train_cond_feat_array = zeros(size(train_df_cond_feat,1),0)
        test_cond_feat_array = zeros(size(test_df_cond_feat,1),0)
    else
        train_cond_feat_array = Tables.matrix(train_df_cond_feat[:, 2:end])
        test_cond_feat_array = Tables.matrix(test_df_cond_feat[:, 2:end])
    end

    train_pred_feat_array = Tables.matrix(train_df_pred_feat[:, 2:end])
    test_pred_feat_array = Tables.matrix(test_df_pred_feat[:, 2:end])

    # Optimizers
    opt_discr = ADAM(lr_discr, (0.5, 0.99))
    opt_gen = ADAM(lr_gen, (0.5, 0.99))

    ## To track losses
    _, num_batch = batch_indexes(df_pred_feat, batch_size, history_time)
    loss_train_step = zeros(Float32, epochs*num_batch, 2)
    loss_train_epoch = zeros(Float32, epochs, 2)
    loss_test_epoch = zeros(Float32, epochs, features_pred)
    min_test_loss = 10 ## Set an arbitary loss value to start saving the model once the loss is less than this
    min_epoch = []
    ## Train
    train_steps = 0
    for i in 1:epochs
        indexes, num_batch = batch_indexes(train_df_pred_feat, batch_size, history_time)
        for j in 1:num_batch
            noise = randn(Float32, noise_dim, batch_size)*noise_sd
            cond_feat_batch, pred_feat_batch = create_batch(indexes, j, seq_len, train_cond_feat_array, train_pred_feat_array)
            loss = train_step(cond_feat_batch, pred_feat_batch, noise, gen, discr, opt_gen, opt_discr, w_adv, w_pred)
            loss_train_step[train_steps + 1, 1] = loss["discr"]*1f0
            loss_train_step[train_steps + 1, 2] = loss["gen"]*1f0

            if train_steps % info_freq == 0
                @info("Training Step ($train_steps): Discr Loss = $(loss["discr"]), Gen Loss = $(loss["gen"])")
            end
            train_steps += 1
        end
        ep_range = range((i-1)*num_batch + 1, i*num_batch)
        loss_train_epoch[i, 1] = mean(loss_train_step[ep_range, 1])
        loss_train_epoch[i, 2] = mean(loss_train_step[ep_range, 2])
        loss_test_epoch[i, :] = test_loss(gen, test_cond_feat_array, test_pred_feat_array, seq_len, test_samp_size, noise_dim, noise_sd)
        
        ## Test to see if test loss is the best. If so save the model
        loss_mean = mean(loss_test_epoch[i,:])
        if loss_mean < min_test_loss
            min_test_loss = loss_mean

            params_gen_g_common, params_gen_g_cond = Flux.params(gen.g_common), Flux.params(gen.g_cond)
            @save "$(model_name)_gen_g_common_min_epoch.bson" params_gen_g_common
            @save "$(model_name)_gen_g_cond_min_epoch.bson" params_gen_g_cond


            params_discr_d_cond_feat, params_discr_d_pred_feat, params_discr_d_common = Flux.params(discr.d_cond_feat), Flux.params(discr.d_pred_feat), Flux.params(discr.d_common)
            @save "$(model_name)_discr_d_cond_feat_min_epoch.bson" params_discr_d_cond_feat
            @save "$(model_name)_discr_d_pred_feat_min_epoch.bson" params_discr_d_pred_feat
            @save "$(model_name)_discr_d_common_min_epoch.bson" params_discr_d_common
            
            min_epoch = i
        end 

        ## Also Save every save_model_freq epoch

        if i % save_model_freq == 0

            params_gen_g_common, params_gen_g_cond = Flux.params(gen.g_common), Flux.params(gen.g_cond)
            @save "$(model_name)_gen_g_common_epoch_$(i).bson" params_gen_g_common
            @save "$(model_name)_gen_g_cond_epoch_$(i).bson" params_gen_g_cond

            params_discr_d_cond_feat, params_discr_d_pred_feat, params_discr_d_common = Flux.params(discr.d_cond_feat), Flux.params(discr.d_pred_feat), Flux.params(discr.d_common)
            @save "$(model_name)_discr_d_cond_feat_epoch_$(i).bson" params_discr_d_cond_feat
            @save "$(model_name)_discr_d_pred_feat_epoch_$(i).bson" params_discr_d_pred_feat
            @save "$(model_name)_discr_d_common_epoch_$(i).bson" params_discr_d_common
        end
    end
    return loss_train_step, loss_train_epoch, loss_test_epoch, gen, discr, min_epoch
end


"""
## epoch is either the integer of epoch number OR the string "min" if want epoch with lowest test loss
"""
function load_discr(model_name, epoch, seq_len, features_cond, features_pred, hyparams)
    if epoch == "min" 
        epoch_tail = "min_epoch"
    else
        epoch_tail = "epoch_$(epoch)"
    end
    params_files = ["$(model_name)_discr_d_common_$(epoch_tail).bson", "$(model_name)_discr_d_cond_feat_$(epoch_tail).bson", "$(model_name)_discr_d_pred_feat_$(epoch_tail).bson"]

    file_common = params_files[1]
    file_cond_feat = params_files[2]
    file_pred_feat = params_files[3]
      
    new_discr = discriminator(seq_len, features_cond, features_pred,hyparams.discr_rnn_layer, hyparams.discr_rnn_units, hyparams.discr_dense_units) 

    params_discr_d_common = Flux.params(new_discr.d_common)
    params_discr_d_cond_feat = Flux.params(new_discr.d_cond_feat)
    params_discr_d_pred_feat = Flux.params(new_discr.d_pred_feat)

    @load file_common params_discr_d_common
    @load file_cond_feat params_discr_d_cond_feat
    @load file_pred_feat params_discr_d_pred_feat

    Flux.loadparams!(new_discr.d_common, params_discr_d_common)
    Flux.loadparams!(new_discr.d_cond_feat, params_discr_d_cond_feat)
    Flux.loadparams!(new_discr.d_pred_feat, params_discr_d_pred_feat)

    return new_discr
end

"""
## epoch is either the integer of epoch number OR the string "min" if want epoch with lowest test loss
"""
function load_gen(model_name, epoch, seq_len, features_cond, features_pred, hyparams)
    
    if epoch == "min" 
        epoch_tail = "min_epoch"
    else
        epoch_tail = "epoch_$(epoch)"
    end

    params_files = ["$(model_name)_gen_g_common_$(epoch_tail).bson", "$(model_name)_gen_g_cond_$(epoch_tail).bson"]

    input_features = features_cond + features_pred

    file_common = params_files[1]
    file_cond = params_files[2]
      
    new_gen = generator(seq_len, input_features, features_pred, hyparams.noise_dim,hyparams.gen_rnn_layer ,hyparams.gen_rnn_units, hyparams.gen_dense_units)
    params_gen_g_common = Flux.params(new_gen.g_common)
    params_gen_g_cond = Flux.params(new_gen.g_cond)

    @load file_common params_gen_g_common
    @load file_cond params_gen_g_cond

    Flux.loadparams!(new_gen.g_common, params_gen_g_common)
    Flux.loadparams!(new_gen.g_cond, params_gen_g_cond)

    return new_gen
end