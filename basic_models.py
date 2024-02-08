import copy
import time

import keras
import numpy as np
import random
import torch
import torch.nn as nn

from statsforecast.models import AutoARIMA, AutoTheta

from other_models.NBEATS import NBeatsNet
from other_models.TimesNet import Model as TimesNet
from other_models.DLinear import Model as DLinear
from other_models.FEDformer import Model as FEDformer
from other_models.Nonstationary_Transformer import Model as Nonstationary_Transformer
from other_models.train_model import train_model, predict, _acquire_device
from SHIFT import SHIFT
from utility import merge_train_val

# Naive forecast
def predict_naive(dataset):
    total_inference_time = 0
    t0 = time.time()
    y_pred = np.array([np.full(dataset["y_train"].shape[-1], X[-1]) for X in dataset["X_test"]])
    total_inference_time += time.time() - t0
    return y_pred, total_inference_time

#=======================================================================
#====================Statistical forecasting methods====================
#=======================================================================

# Automatically optimized model from the theta family; https://www.researchgate.net/publication/223049702_The_theta_model_A_decomposition_approach_to_forecasting
def predict_theta(dataset):
    horizon = len(dataset["y_train"][0])
    total_inference_time = 0
    y_pred = []
    for ts in dataset["X_test"]:
        model = AutoTheta()
        model.fit(ts)

        t0 = time.time()
        pred = model.predict(horizon)["mean"]
        total_inference_time += time.time() - t0

        y_pred.append(pred)

    return np.array(y_pred,dtype=object), total_inference_time

# Automatically optimized ARIMA; https://www.jstor.org/stable/pdf/2284333.pdf
def predict_arima(dataset):
    horizon = len(dataset["y_train"][0])
    total_inference_time = 0
    y_pred = []
    if len(dataset["X_test"][0]) <= 5:
        print("Fallback to naive prediction for short backhorizons")
        return predict_naive(dataset)
    for ts in dataset["X_test"]:
        model = AutoARIMA()
        try:
            model.fit(ts)

            t0 = time.time()
            pred = model.predict(horizon)["mean"]
            total_inference_time += time.time() - t0

            y_pred.append(pred)
        except ZeroDivisionError:
            print("AutoARIMA bug")
            y_pred.append(np.full(horizon, np.median(ts)))

    return np.array(y_pred,dtype=object), total_inference_time

def predict_shift(dataset):
    # If validation data is present, merge this into the training data
    if len(dataset["X_val"]) > 0:
        data = merge_train_val(dataset)
    else:
        data = copy.deepcopy(dataset)
        
    total_inference_time = 0

    y_pred = []
    for index in range(len(data["ts_raw"])):
        h = len(data["y_train"][index][0])
        timeseries = data["ts_raw"][index]

        model = SHIFT()
        model.optimize_hyperparameters(timeseries, data["X_train"][index], data["y_train"][index])

        t0 = time.time()
        pred = model.fit_predict(timeseries, data["X_test"][index], h)
        total_inference_time += time.time() - t0

        y_pred.append(pred)

    return np.array(y_pred, dtype=object), total_inference_time

def predict_shift_ablation(dataset, k=False, fallback=None, chain=None, z_normalization=None):
    # If validation data is present, merge this into the training data
    if len(dataset["X_val"]) > 0:
        data = merge_train_val(dataset)
    else:
        data = copy.deepcopy(dataset)
        
    model = SHIFT()
    if type(k) != bool:
        model.k = k
    if chain != None:
        model.chain = chain
    if z_normalization != None:
        model.z_normalization = z_normalization
    if fallback != None:
        model.fallback = fallback

    y_pred = []
    for index in range(len(data["ts_raw"])):
        h = len(data["y_train"][index][0])
        timeseries = data["ts_raw"][index]

        model.optimize_hyperparameters(timeseries, data["X_train"][index], data["y_train"][index])
        pred = model.fit_predict(timeseries, data["X_test"][index], h)
        y_pred.append(pred)

    return np.array(y_pred, dtype=object)

#=======================================================================
#=========================Deep learning methods=========================
#=======================================================================

def train_nbeats(X_train, y_train, X_val, y_val, random_state=42):
    keras.utils.set_random_seed(int(random_state))
    model = NBeatsNet(
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        forecast_length=y_train.shape[1],
        backcast_length=X_train.shape[1]
    )
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss="mae")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )

    return model

# N-BEATS; https://arxiv.org/pdf/1905.10437
def predict_nbeats(dataset, random_states):
    all_pred = []
    all_inference_times = []
    for r in random_states:
        nbeats = train_nbeats(dataset["X_train"], dataset["y_train"], dataset["X_val"], dataset["y_val"], random_state=r)
        total_inference_time = 0

        t0 = time.time()
        y_pred = nbeats.predict(dataset["X_test"], verbose=0)
        total_inference_time += time.time() - t0
        all_inference_times.append(total_inference_time)

        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)

def train_nbeats_i(X_train, y_train, X_val, y_val, random_state=42):
    keras.utils.set_random_seed(int(random_state))
    model = NBeatsNet(
        stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
        forecast_length=y_train.shape[1],
        backcast_length=X_train.shape[1]
    )
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss="mae")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )

    return model

# GRU; https://arxiv.org/pdf/1412.3555)
def predict_gru(dataset, random_states):
    def train_gru(X_train, y_train, X_val, y_val, random_state=42):
        keras.utils.set_random_seed(int(random_state))

        inputs = keras.Input(shape=(X_train.shape[1],1))
        gru = keras.layers.GRU(100, activation="tanh", return_sequences=True)(inputs)
        gru2 = keras.layers.GRU(100, activation="tanh", return_sequences=False)(gru)
        dense = keras.layers.Dense(y_train.shape[1], activation="linear")(gru2)
        model = keras.Model(inputs=inputs, outputs=dense)

        optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
        model.compile(optimizer, loss="mae")

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=128,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0,
        )

        return model
    
    all_pred = []
    all_inference_times = []
    for r in random_states:
        gru = train_gru(dataset["X_train"], dataset["y_train"], dataset["X_val"], dataset["y_val"], random_state=r)
        X_test = dataset["X_test"].reshape(dataset["X_test"].shape[0], dataset["X_test"].shape[1], 1)
        total_inference_time = 0

        t0 = time.time()
        y_pred = gru.predict(X_test, verbose=0)
        total_inference_time += time.time() - t0
        all_inference_times.append(total_inference_time)

        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)

# DLinear; https://arxiv.org/pdf/2205.13504
def predict_dlinear(dataset, frequency_map, random_states):
    class ModelConfig(object):
        pass
    config = ModelConfig()
    
    # config.use_gpu = True
    config.use_gpu = False
    config.devices = "0,1,2,3"
    config.gpu = 0
    # config.use_multi_gpu = True
    config.use_multi_gpu = False

    config.pred_len = dataset["y_train"].shape[1]
    config.label_len = dataset["y_train"].shape[1]
    config.seq_len = dataset["X_train"].shape[1]
    config.enc_in = 1
    config.embed = "timeF"
    config.freq = "h"
    config.frequency_map = frequency_map
    config.patience = 3
    config.learning_rate = 0.001
    config.train_epochs = 10
    config.features = "M"
    config.lradj = "type1"
    config.moving_avg = 25
    config.batch_size = 16

    all_pred = []
    all_inference_times = []
    for r in random_states:
        random.seed(r)
        torch.manual_seed(r)
        np.random.seed(r)

        # Create model with pytorch
        device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)

        model = DLinear(config).float()
        if config.use_gpu:
            device_ids = config.devices.split(',')
            device_ids = [int(id_) for id_ in device_ids]
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_val = dataset["X_val"]
        y_val = dataset["y_val"]
        X_test = dataset["X_test"]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Train the model with early stopping based on validation data
        model = train_model(model, config, X_train, y_train, X_val, y_val)

        total_inference_time = 0

        t0 = time.time()
        y_pred = predict(model, config, X_test)
        total_inference_time += time.time() - t0
        all_inference_times.append(total_inference_time)

        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)


# FEDformer; https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf
def predict_fedformer(dataset, frequency_map, d_ff, d_model, random_states):
    class ModelConfig(object):
        pass
    config = ModelConfig()
    
    # config.use_gpu = True
    config.use_gpu = False
    config.devices = "0,1,2,3"
    config.gpu = 0
    # config.use_multi_gpu = True
    config.use_multi_gpu = False

    config.pred_len = dataset["y_train"].shape[1]
    config.label_len = dataset["y_train"].shape[1]
    config.seq_len = dataset["X_train"].shape[1]
    config.e_layers = 2
    config.enc_in = 1
    config.embed = "timeF"
    config.dropout = 0.1
    config.c_out = 1
    config.d_model = d_model
    config.d_ff = d_ff
    config.freq = "h"
    config.frequency_map = frequency_map
    config.patience = 3
    config.learning_rate = 0.001
    config.train_epochs = 5
    config.features = "M"
    config.lradj = "type1"
    config.top_k = 5
    config.num_kernels = 6
    config.batch_size = 16

    config.moving_avg = 25
    config.dec_in = 1
    config.n_heads = 8
    config.activation = "gelu"
    config.d_layers = 1

    all_pred = []
    all_inference_times = []
    for r in random_states:
        random.seed(r)
        torch.manual_seed(r)
        np.random.seed(r)

        # Create model with pytorch
        device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)

        model = FEDformer(config).float()
        if config.use_gpu:
            device_ids = config.devices.split(',')
            device_ids = [int(id_) for id_ in device_ids]
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_val = dataset["X_val"]
        y_val = dataset["y_val"]
        X_test = dataset["X_test"]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Train the model with early stopping based on validation data
        model = train_model(model, config, X_train, y_train, X_val, y_val)

        total_inference_time = 0

        t0 = time.time()
        y_pred = predict(model, config, X_test)
        total_inference_time += time.time() - t0
        all_inference_times.append(total_inference_time)

        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)

# Non-stationary transformer; https://proceedings.neurips.cc/paper_files/paper/2022/file/4054556fcaa934b0bf76da52cf4f92cb-Paper-Conference.pdf
def predict_nonstationary_transformer(dataset, frequency_map, d_ff, d_model, random_states):
    class ModelConfig(object):
        pass
    config = ModelConfig()
    
    # config.use_gpu = True
    config.use_gpu = False
    config.devices = "0,1,2,3"
    config.gpu = 0
    # config.use_multi_gpu = True
    config.use_multi_gpu = False

    config.pred_len = dataset["y_train"].shape[1]
    config.label_len = dataset["y_train"].shape[1]
    config.seq_len = dataset["X_train"].shape[1]
    config.e_layers = 2
    config.enc_in = 1
    config.embed = "timeF"
    config.dropout = 0.1
    config.c_out = 1
    config.d_model = d_model
    config.d_ff = d_ff
    config.freq = "h"
    config.frequency_map = frequency_map
    config.patience = 3
    config.learning_rate = 0.001
    config.train_epochs = 5
    config.features = "M"
    config.lradj = "type1"
    config.top_k = 5
    config.num_kernels = 6
    config.batch_size = 16

    config.dec_in = 1
    config.n_heads = 8
    config.activation = "gelu"
    config.d_layers = 1
    config.output_attention = True
    config.factor = 1
    config.p_hidden_dims = [256,256]
    config.p_hidden_layers = 2

    all_pred = []
    all_inference_times = []
    for r in random_states:
        random.seed(r)
        torch.manual_seed(r)
        np.random.seed(r)

        # Create model with pytorch
        device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)

        model = Nonstationary_Transformer(config).float()
        if config.use_gpu:
            device_ids = config.devices.split(',')
            device_ids = [int(id_) for id_ in device_ids]
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_val = dataset["X_val"]
        y_val = dataset["y_val"]
        X_test = dataset["X_test"]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Train the model with early stopping based on validation data
        model = train_model(model, config, X_train, y_train, X_val, y_val)

        total_inference_time = 0

        t0 = time.time()
        y_pred = predict(model, config, X_test)
        total_inference_time += time.time() - t0
        all_inference_times.append(total_inference_time)

        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)

# TimesNet; https://openreview.net/pdf?id=ju_Uqw384Oq
def predict_timesnet(dataset, frequency_map, d_ff, d_model, random_states):
    class ModelConfig(object):
        pass
    config = ModelConfig()

    config.use_gpu = False
    config.devices = "0,1,2,3"
    config.gpu = 0
    config.use_multi_gpu = False

    config.pred_len = dataset["y_train"].shape[1]
    config.label_len = dataset["y_train"].shape[1]
    config.seq_len = dataset["X_train"].shape[1]
    config.e_layers = 2
    config.enc_in = 1
    config.embed = "timeF"
    config.dropout = 0.1
    config.c_out = 1
    config.d_model = d_model
    config.d_ff = d_ff
    config.freq = "h"
    config.frequency_map = frequency_map
    config.patience = 3
    config.learning_rate = 0.001
    config.train_epochs = 5
    config.features = "M"
    config.lradj = "type1"
    config.top_k = 5
    config.num_kernels = 6
    config.batch_size = 16

    all_pred = []
    all_inference_times = []
    for r in random_states:
        random.seed(r)
        torch.manual_seed(r)
        np.random.seed(r)

        # Create model with pytorch
        device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)

        model = TimesNet(config).float()
        if config.use_gpu:
            device_ids = config.devices.split(',')
            device_ids = [int(id_) for id_ in device_ids]
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_val = dataset["X_val"]
        y_val = dataset["y_val"]
        X_test = dataset["X_test"]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Train the model with early stopping based on validation data
        model = train_model(model, config, X_train, y_train, X_val, y_val)

        total_inference_time = 0

        t0 = time.time()
        y_pred = predict(model, config, X_test)
        total_inference_time += time.time() - t0
        all_inference_times.append(total_inference_time)

        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)