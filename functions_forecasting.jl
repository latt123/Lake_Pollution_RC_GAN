# # Functions for forecasting and viewing distrubutions


"""
## 'Samp_size' forecasts for each value of the test set used for training
Inputs:
- timeseries - Data that was used for train() function (including test set)
- hyparams - Hyperparameters used for train() function
- gen - Generator function (output of train() function)
- samp_size - the number of samples to be generated for each point in loss_test_step

Outputs:
- ts_and_forecasts - array of size (length of timeseries,features of timeseries,samp_size) where each matrix (:,:,i) is the part of data set that was used for training combined with generated values in place of the test set.
- forecast_len - the length the forecast part of the outputed timeseries (also lenght of training set)
"""
function forecast(df_cond_feat, df_pred_feat, hyparams, gen, samp_size , start_forecast, forecast_time)
    ## Determing date and index of start of forecast in data frame
    tbm = df_pred_feat.Date[2] - df_pred_feat.Date[1]
    start_in_df = start_forecast
    index_start_in_df = range(1,size(df_pred_feat,1))[df_pred_feat.Date .== start_in_df][1]

    history_time = hyparams.history_time
    seq_len = Int.(history_time/tbm)

    noise_dim = hyparams.noise_dim
    noise_sd = hyparams.noise_sd
    len = size(df_cond_feat,1)

    forecast_len = Int.(forecast_time/tbm)

    features_cond = size(df_cond_feat, 2) - 1 
    features_pred = size(df_pred_feat, 2) - 1 

    past_and_forecast_pred_feat = zeros(Float32, features_pred, samp_size, index_start_in_df + forecast_len)
    for j in 1:samp_size
        past_and_forecast_pred_feat[:,j,range(1,index_start_in_df)] = permutedims(Tables.matrix(df_pred_feat[1:index_start_in_df,2:end]),(2,1))
        for  i in 1:forecast_len
            prev_seq_pred_feat_reshape = past_and_forecast_pred_feat[:,j:j,range(index_start_in_df-seq_len+i,index_start_in_df-1+i)]
            if size(df_cond_feat,2) == 1 
                prev_seq_cond_feat_reshape = zeros(Float32, 0, 1, seq_len) 
            else
                prev_seq_cond_feat = permutedims(Tables.matrix(df_cond_feat[range(index_start_in_df-seq_len+i,index_start_in_df-1+i),2:end]),(2,1))
                prev_seq_cond_feat_reshape = reshape(prev_seq_cond_feat, features_cond, 1, seq_len)
            end
            noise = randn(Float32,noise_dim,1)*noise_sd
            Flux.reset!(gen.g_cond)
            Flux.reset!(gen.g_common)
            next_val = gen(prev_seq_cond_feat_reshape, prev_seq_pred_feat_reshape, noise)
            past_and_forecast_pred_feat[:,j,index_start_in_df+i] = next_val
        end
    end
    return past_and_forecast_pred_feat, forecast_len
end


function forecast_2(df_cond_feat, df_pred_feat, hyparams, gen, samp_size , start_forecast, forecast_time, n_before)
    ## Determing date and index of start of forecast in data frame
    tbm = df_pred_feat.Date[2] - df_pred_feat.Date[1]
    start_in_df = @subset(df_pred_feat, start_forecast - tbm .<= :Date .< start_forecast).Date[1]
    index_start_in_df = range(1,size(df_pred_feat,1))[df_pred_feat.Date .== start_in_df][1]

    history_time = hyparams.history_time
    seq_len = Int.(history_time/tbm)

    noise_dim = hyparams.noise_dim
    noise_sd = hyparams.noise_sd
    len = size(df_cond_feat,1)

    forecast_len = Int.(forecast_time/tbm)

    features_cond = size(df_cond_feat, 2) - 1 
    features_pred = size(df_pred_feat, 2) - 1 

    past_and_forecast_pred_feat = zeros(Float32, features_pred, samp_size, index_start_in_df + forecast_len)
    for j in 1:samp_size
        past_and_forecast_pred_feat[:,j,range(1,index_start_in_df)] = permutedims(Tables.matrix(df_pred_feat[1:index_start_in_df,2:end]),(2,1))
        for  i in 1:(forecast_len+n_before)
            prev_seq_pred_feat_reshape = past_and_forecast_pred_feat[:,j:j,range(index_start_in_df-seq_len+i-n_before,index_start_in_df-1+i-n_before)]
            if size(df_cond_feat,2) == 1 
                prev_seq_cond_feat_reshape = zeros(Float32, 0, 1, seq_len) 
            else
                prev_seq_cond_feat = permutedims(Tables.matrix(df_cond_feat[range(index_start_in_df-seq_len+i-n_before,index_start_in_df-1+i-n_before),2:end]),(2,1))
                prev_seq_cond_feat_reshape = reshape(prev_seq_cond_feat, features_cond, 1, seq_len)
            end
            noise = randn(Float32,noise_dim,1)*noise_sd
            # Flux.reset!(gen.g_cond)
            # Flux.reset!(gen.g_common)
            next_val = gen(prev_seq_cond_feat_reshape, prev_seq_pred_feat_reshape, noise)
            if i > n_before
                past_and_forecast_pred_feat[:,j,index_start_in_df+i-n_before] = next_val
            end
        end
    end

    return past_and_forecast_pred_feat, forecast_len
end


"""
## Plot History and forecast Quantiles of length 'forecast_len' for feature 'featuring'.
Inputs:
- data - orginal time series inputed into train() function
- ts_and_forecasts - forecasts as outputed from forecast_test() function
- quantiles - quantiles to be plotted (3 Quantiles asscending)
- featuring - feature to be plotted
- forecast_len - length of forecast part of ts_and_forecast as outputed from forecast_test() function

Output:
- Plot of timeseries and forecast of test part of data set for feature 'featuring'.
"""
function plot_forecast(data,ts_and_forecast,quantiles,featuring,forecast_len) 
    x = ts_and_forecast
    x_len = size(ts_and_forecast,1)
    data_len = x_len - forecast_len
    f(x,q) = [quantile(c, q) for c in eachslice(x, dims = 1)]
    f_median = f(x[:,featuring,:],1/2)      

    forecast_range  = range(data_len,x_len)
    rb_1_low = (f_median - f(x[:,featuring,:],quantiles[1]))[forecast_range]
    rb_1_up =  (f(x[:,featuring,:],1-quantiles[1]) - f_median)[forecast_range]
    rb_2_low = (f_median - f(x[:,featuring,:],quantiles[2]))[forecast_range]
    rb_2_up =  (f(x[:,featuring,:],1-quantiles[2]) - f_median)[forecast_range]
    rb_3_low = (f_median - f(x[:,featuring,:],quantiles[3]))[forecast_range]
    rb_3_up =  (f(x[:,featuring,:],1-quantiles[3]) - f_median)[forecast_range]

    plot(range(1,data_len),f_median[1:data_len],label="Real time series",title="feature $(featuring)",legend=:outertopright,linecolor=[:black])
    plot!(forecast_range ,data[data_len:x_len,featuring],linestyle=:dash,linecolor=[:black],label=false)
    plot!(forecast_range ,f_median[data_len:x_len],ribbon = (rb_1_low,rb_1_up),label="Quantile $(quantiles[1])",title="feature $(featuring)",legend=:outertopright)
    plot!(forecast_range ,f_median[data_len:x_len],ribbon = (rb_2_low,rb_2_up),label="Quantile $(quantiles[2])",title="feature $(featuring)",legend=:outertopright)
    plot!(forecast_range ,f_median[data_len:x_len],ribbon = (rb_3_low,rb_3_up),label="Quantile $(quantiles[3])",title="feature $(featuring)",legend=:outertopright)
end


"""
##  Plots distrubution prediction of 'point' value after the forecast starts for feature 'featuring'
Inputs:
- point - point after forecast to produce density of 
- df_pred_feat - 'Real' data set from which model is trained and prediction is trying to mimic
- prediction - array of a forecast
- start_forecast - data for which the forecast will start after
- featuring - which feature to produce density of

Outputs
Plot of density with line for actual value from data set. 
"""
function dist_pred_point(point,df_pred_feat,prediction,start_forecast,featuring)
    next_time_pred_index = findall(df_pred_feat.Date .== start_forecast)[1]
    dist = prediction[next_time_pred_index+point,featuring,:]
    d_pred = [density(dist,label=false,title="$(df_pred_feat.Date[next_time_pred_index+point])",titlefontsize=8)
    vline!([df_pred_feat[next_time_pred_index+point,featuring+1]],linestyle=:dash,linewidth = 2,label=false)]
    return d_pred
end


"""
For a sequence ending in a index 'index_discr' in 'df_pred', 
returns the probability judged by the 'discriminator' that for each value in 'x_range',
that if that value followed the sequence then it is from the real data set.
"""
function discr_x_range(discr, x_range, variable, df_pred, pred_features, Δt_sparse, hyparams, index_discr)
    
    # Set parameters
    min_range = min(x_range...)
    max_range = max(x_range...)
    x_values = size(x_range)[1]
    seq_len = Int.((hyparams.history_time/Δt_sparse))
    range_good_indexes = Int.(seq_len+1):(size(df_pred,1)-1)
    featuring = range(1,size(pred_features,1))[pred_features.==variable][1]

    # Conditional array input into discr()
    if size(df_cond,2) == 1
        icf_repeat = zeros(Float32,0,x_values,seq_len)
    else
        icf = permutedims(Tables.matrix(df_cond[range_good_indexes[index_discr]-seq_len+1:range_good_indexes[index_discr],2:end]),(2,1))
        icf_repeat = permutedims(repeat(icf,1,1,x_values), (1,3,2))
    end
    # Prediction array input into discr()
    ipf = permutedims(Tables.matrix(df_pred[range_good_indexes[index_discr]-seq_len+1:range_good_indexes[index_discr],2:end]),(2,1))
    ipf_repeat = permutedims(repeat(ipf,1,1,x_values),(1,3,2))

    # Post prediction array input into discr()
    ppf = permutedims(Tables.matrix(df_pred[range_good_indexes[index_discr]+1:range_good_indexes[index_discr]+1,2:end]),(2,1))
    ppf_repeat = permutedims(repeat(ppf,1,1,x_values),(1,3,2))
    ppf_repeat[featuring,:,1] = range(min_range, max_range,x_values)

    # Reset memory paramters in discr
    Flux.reset!(discr.d_cond_feat)
    Flux.reset!(discr.d_pred_feat)
    Flux.reset!(discr.d_common)

    # Input values into discriminator 
    prob_real_fixed = discr(icf_repeat,ipf_repeat,ppf_repeat)[1,:]

    # save the actual value in data set
    real_post = ppf[featuring]

    return prob_real_fixed, real_post
end

compute_quantile(x,q; dims = 1) = [quantile(c, q) for c in eachslice(x, dims = dims)]

"""
## Plot History and forecast Quantiles of length 'forecast_len' for feature 'featuring'.
Inputs:
- df_original - orginal time series inputed into train() function
- ts_and_forecasts - forecasts as outputed from forecast_test() function
- quantiles - quantiles to be plotted (3 Quantiles asscending)
- featuring - feature to be plotted
- forecast_len - length of forecast part of forecast_array_denorm as outputed from forecast function

Output:
- Plot of timeseries and forecast of test part of df_original set for feature 'featuring'.
"""
function plot_forecast(df_original,forecast_array_denorm,quantiles, featuring_name, forecast_len) 

    pred_features = propertynames(df_original)[2:end]
    featuring = range(1,size(pred_features,1))[pred_features.==featuring_name][1]

    x = forecast_array_denorm
    x_len = size(x,1)
    data_len = x_len - forecast_len
    f(x,q) = [quantile(c, q) for c in eachslice(x, dims = 1)]
    f_median = f(x[:,featuring,:],1/2)      

    forecast_range  = range(data_len, x_len)
    rb_1_low = (f_median - f(x[:,featuring,:],quantiles[1]))[forecast_range]
    rb_1_up =  (f(x[:,featuring,:],1-quantiles[1]) - f_median)[forecast_range]
    rb_2_low = (f_median - f(x[:,featuring,:],quantiles[2]))[forecast_range]
    rb_2_up =  (f(x[:,featuring,:],1-quantiles[2]) - f_median)[forecast_range]
    rb_3_low = (f_median - f(x[:,featuring,:],quantiles[3]))[forecast_range]
    rb_3_up =  (f(x[:,featuring,:],1-quantiles[3]) - f_median)[forecast_range]

    plot(df_original.Date[range(1,data_len)],f_median[1:data_len],label="Real time series",title=featuring_name,legend=:topright,linecolor=[:black])
    plot!(df_original.Date[forecast_range] ,f_median[data_len:x_len],ribbon = (rb_1_low,rb_1_up),label="Quantile $(quantiles[1])")
    plot!(df_original.Date[forecast_range],f_median[data_len:x_len],ribbon = (rb_2_low,rb_2_up),label="Quantile $(quantiles[2])")
    plot!(df_original.Date[forecast_range] ,f_median[data_len:x_len],ribbon = (rb_3_low,rb_3_up),label="Quantile $(quantiles[3])")
    plot!(df_original.Date[range(data_len, x_len)] ,df_original[range(data_len, x_len),featuring+1],linecolor=[:black],label=false)
end

## Function that inputs generator and outputs forecast image (combine the steps above)



"""
## Generate a gif for a specified variable that has index 'variable_index' in the list of variables 'prediction_variables'
"""
function gen_forecast_gif(variable_index, gen_gif_list, epoch, samp_size, forecast_start, forecast_length, df_cond, df_pred, df_pred_sparse, hyparams, prop_names, prediction_variables, quantiles)
    gen = gen_gif_list[epoch]
    past_and_forecast_pred_feat, forecast_len = forecast(df_cond, df_pred, hyparams, gen, samp_size , forecast_start, forecast_length)
    fcast = permutedims(past_and_forecast_pred_feat,(3,1,2))
    forecast_array_denorm = copy(fcast)
    for j in 1:samp_size    
        reshape_forecast = fcast[:,:,j]
        df_forecast = DataFrame(Date = df_pred.Date[1:size(reshape_forecast,1)])
        k=1
        for i in 1:size(prediction_variables,1)
            index_pred = range(1,size(prop_names,1))[prop_names .== prediction_variables[i]][1]
            df_forecast[!, prediction_variables[i]] = Vector{Float32}(reshape_forecast[:,k])
            k = k + 1
        end
        denorm_forecast = denormalise(df_pred_sparse, df_forecast)
        forecast_array_denorm[:,:,j] = Tables.matrix(denorm_forecast[:,(2:end)])
    end
    plot_feat_1 = plot_forecast(df_pred_sparse, forecast_array_denorm,quantiles, prediction_variables[variable_index], forecast_len)
    plot(plot_feat_1,size=(600,400), legend = :topleft)
end        
