import copy

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def flatten(dataset):
    flattened = copy.deepcopy(dataset)
    original_shapes = {}
    for k in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        original_shapes[k] = [flattened[k][i].shape for i in range(len(flattened[k]))]
        flattened[k] = np.array([i for l in flattened[k] for i in l])
    return flattened, original_shapes

def inverse_flatten(dataset, original_shapes):
    inversed = copy.deepcopy(dataset)
    for k, shapes in original_shapes.items():
        inversed_values = []
        i = 0
        for shape in shapes:
            inversed_values.append(inversed[k][i:i+shape[0]])
            i += shape[0]
        inversed[k] = inversed_values
    return inversed

def inverse_flatten_predictions(predictions, original_shapes):
    y_test_shapes = original_shapes["y_test"]
    inversed_predictions = []
    i = 0
    for shape in y_test_shapes:
        inversed_predictions.append(predictions[i:i+shape[0]])
        i += shape[0]
    return np.array(inversed_predictions, dtype=object)

def scale(dataset):
    def scale_feature(feature, scaler):
        original_shape = feature.shape
        feature = np.array([i for l in feature for i in l])
        transformed = remove_dimension(scaler.transform(add_dimension(feature)))
        return transformed.reshape(original_shape).astype(float)

    scalers = []
    scaled = copy.deepcopy(dataset)
    for i in range(len(scaled["ts_raw"])):
        scaler = StandardScaler()
        # scaler = MinMaxScaler((0,1))
        scaled["ts_raw"][i] = remove_dimension(scaler.fit_transform(add_dimension(scaled["ts_raw"][i])))
        scaled["ts_full"][i] = remove_dimension(scaler.transform(add_dimension(scaled["ts_full"][i])))
        scaled["X_train"][i] = scale_feature(scaled["X_train"][i], scaler)
        scaled["y_train"][i] = scale_feature(scaled["y_train"][i], scaler)
        scaled["X_test"][i] = scale_feature(scaled["X_test"][i], scaler)
        scaled["y_test"][i] = scale_feature(scaled["y_test"][i], scaler)
        if len(scaled["X_val"][i]) > 0:
            scaled["X_val"][i] = scale_feature(scaled["X_val"][i], scaler)
            scaled["y_val"][i] = scale_feature(scaled["y_val"][i], scaler)
        scalers.append(scaler)
    return scaled, scalers

def inverse_scale_feature(feature, scaler):
    feature = np.array(feature)
    original_shape = feature.shape
    while len(feature.shape) > 1:
        feature = np.array([i for l in feature for i in l])
    transformed = remove_dimension(scaler.inverse_transform(add_dimension(feature)))
    return transformed.reshape(original_shape)

def inverse_scale(dataset, scalers):
    inversed = copy.deepcopy(dataset)
    for i in range(len(inversed["ts_raw"])):
        scaler = scalers[i]

        inversed["ts_raw"][i] = remove_dimension(scaler.inverse_transform(add_dimension(inversed["ts_raw"][i])))
        inversed["ts_full"][i] = remove_dimension(scaler.inverse_transform(add_dimension(inversed["ts_full"][i])))

        inversed["X_train"][i] = inverse_scale_feature(inversed["X_train"][i], scaler)
        inversed["y_train"][i] = inverse_scale_feature(inversed["y_train"][i], scaler)
        inversed["X_test"][i] = inverse_scale_feature(inversed["X_test"][i], scaler)
        inversed["y_test"][i] = inverse_scale_feature(inversed["y_test"][i], scaler)
        if len(inversed["X_val"][i] > 0):
            inversed["X_val"][i] = inverse_scale_feature(inversed["X_val"][i], scaler)
            inversed["y_val"][i] = inverse_scale_feature(inversed["y_val"][i], scaler)

    return inversed

def inverse_scale_predictions(predictions, scalers):
    inversed = []
    for pred, scaler in zip(predictions, scalers):
        inversed.append(inverse_scale_feature(pred, scaler))
    return np.array(inversed, dtype=object)

def add_dimension(array):
    return array.reshape(array.shape + (1,))

def remove_dimension(array):
    return array.reshape(array.shape[:len(array.shape)-1])

def train_test_split(ts, horizon, input_size, train_size, val_size, stride_length):
    def generate_x_y(data, horizon, input_size, stride_length):
        if input_size + horizon > len(data):
            return np.array([]), np.array([])
        x_windows = np.lib.stride_tricks.sliding_window_view(data[:-horizon], input_size)[::stride_length, :]
        y_windows = np.lib.stride_tricks.sliding_window_view(data[input_size:], horizon)[::stride_length, :]
        return x_windows, y_windows

    train_stop = int((train_size + val_size) * len(ts))

    train = ts[:train_stop]
    # Go back only horizon steps to not overlap inputs of test instances and training instances
    test = ts[train_stop-input_size+1:]
    # test = ts[train_stop-horizon:]

    X_train, y_train = generate_x_y(train, horizon, input_size, stride_length)
    X_test, y_test = generate_x_y(test, horizon, input_size, stride_length)
    if len(X_test) == 0 or len(X_train) == 0:
        return {}

    if val_size > 0:
        val_split = int((val_size / (val_size + train_size)) * len(X_train))
        if val_split == 0:
            X_val = np.array([])
            y_val = np.array([])
        else:
            X_val = X_train[-val_split:]
            y_val = y_train[-val_split:]
            X_train = X_train[:-val_split]
            y_train = y_train[:-val_split]
    else:
        X_val = np.array([])
        y_val = np.array([])
    return {
        "ts_raw": np.array(ts[:train_stop]),
        "ts_full": np.array(ts),
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

# ------------------------------------------------
# ------------------ UNIVARIATE ------------------
# ------------------------------------------------

def read_dataset(dataset, train_size=0.6, val_size=0.2, stride_length=1, backhorizon=10, horizon=10):
    data_path = "data/UTS/"

    cif_path = data_path + "cif_2016_dataset.tsf"
    nn5_path = data_path + "nn5_daily_dataset_without_missing_values.tsf"
    tourism_path = data_path + "tourism_monthly_dataset.tsf"
    weather_path = data_path + "weather_prediction_dataset.csv"

    m4_hourly_path = data_path + "Hourly-train.csv"
    m4_weekly_path = data_path + "Weekly-train.csv"
    m4_yearly_path = data_path + "Yearly-train.csv"

    m3_monthly_path = data_path + "M3_monthly_TSTS.csv"
    m3_quarterly_path = data_path + "M3_quarterly_TSTS.csv"
    m3_yearly_path = data_path + "M3_yearly_TSTS.csv"
    m3_other_path = data_path + "M3_other_TSTS.csv"

    transactions_path = data_path + "transactions.csv"

    if dataset == "cif":
        df = read_cif(cif_path)
    elif dataset == "nn5":
        df = read_nn5(nn5_path)
    elif dataset == "tourism":
        df = read_tourism(tourism_path)
    elif dataset == "weather":
        df = read_weather(weather_path)
    elif dataset == "m4_h":
        df = read_m4(m4_hourly_path)
    elif dataset == "m4_w":
        df = read_m4(m4_weekly_path)
    elif dataset == "m4_y":
        df = read_m4(m4_yearly_path)
    elif dataset == "m3_m":
        df = read_m3(m3_monthly_path)
    elif dataset == "m3_q":
        df = read_m3(m3_quarterly_path)
    elif dataset == "m3_y":
        df = read_m3(m3_yearly_path)
    elif dataset == "m3_o":
        df = read_m3(m3_other_path)
    elif dataset == "transactions":
        df = read_transactions(transactions_path)
    else:
        print("Attempting to read unknown dataset")
        raise NotImplementedError()

    train_test = {
        "ts_raw": [],
        "ts_full": [],
        "X_train": [],
        "X_val": [],
        "X_test": [],
        "y_train": [],
        "y_val": [],
        "y_test": [],
    }

    for i in range(len(df)):
        ts = df.iloc[i,:].to_numpy()
        ts = ts[~np.isnan(ts)]

        split_dict = train_test_split(ts, horizon, backhorizon, train_size, val_size, stride_length)
        for k,v in split_dict.items():
            train_test[k].append(v)

    for k,v in train_test.items():
        numpy_converted = np.array(v, dtype=object)
        if len(numpy_converted.shape) == 1:
            train_test[k] = numpy_converted
        else:
            train_test[k] = np.array(v)
    train_test["is_multivariate"] = False
    return train_test

def read_cif(path):
    df = pd.read_csv(
        path,
        sep=":|,",
        encoding="cp1252",
        engine="python",
        header=None,
        index_col=0,
        skiprows=16
    )
    # Filter for 12 months forecasting horizon
    df = df[df.iloc[:, 0] == 12]
    return df.iloc[:, 1:]

def read_nn5(path):
    df = pd.read_csv(
        path,
        sep=":|,",
        engine="python",
        header=None,
        index_col=0,
        skiprows=19
    )
    return df.iloc[:, 1:]

def read_tourism(path):
    df = pd.read_csv(
        path,
        sep=":",
        encoding="cp1252",
        engine="python",
        header=None,
        index_col=0,
        skiprows=15
    )
    df = df.loc[:, 2].str.split(",", expand=True)
    df = df.astype("float")
    return df

def read_weather(path):
    df = pd.read_csv(
        path,
        sep=","
    )
    columns = df.columns
    temperature_columns = columns.str.endswith("temp_mean")
    df = df.loc[:,temperature_columns]
    df = df.T
    return df

def read_m4(path):
    df = pd.read_csv(path)
    df = df.iloc[:,1:]

    return df

def read_m3(path):
    df = pd.read_csv(path)
    df = pd.DataFrame(df.groupby("series_id")["value"])
    df = df.iloc[:,1]
    all_rows = []
    for row in df:
        all_rows.append(np.array(row))
    
    return pd.DataFrame(all_rows)

def read_transactions(path):
    df = pd.read_csv(path)
    df = pd.pivot(df, index=["store_nbr"], columns=["date"], values=["transactions"])
    df.columns = range(df.columns.size)
    all_rows = []
    for _, row in df.iterrows():
        row.dropna(inplace=True)
        all_rows.append(np.array(row))

    return pd.DataFrame(all_rows)


# ------------------------------------------------
# ------------------ MULTIVARIATE ------------------
# ------------------------------------------------

def read_dataset_mts(dataset, train_size=0.6, val_size=0.2, stride_length=1, backhorizon=10, horizon=10):
    data_path = "data/MTS/"

    exchange_rate_path = data_path + "exchange_rate.csv"
    illness_path = data_path + "national_illness.csv"

    ett_h1_path = data_path + "ETTh1.csv"
    ett_h2_path = data_path + "ETTh2.csv"

    if dataset == "exchange_rate":
        df = read_exchange_rate(exchange_rate_path)
    elif dataset == "illness":
        df = read_illness(illness_path)
    elif dataset == "ett_h1":
        df = read_ett(ett_h1_path)
    elif dataset == "ett_h2":
        df = read_ett(ett_h2_path)
    else:
        print("Attempting to read unknown dataset")
        raise NotImplementedError()

    train_test = {
        "ts_raw": [],
        "ts_full": [],
        "X_train": [],
        "X_val": [],
        "X_test": [],
        "y_train": [],
        "y_val": [],
        "y_test": [],
    }
    df = df.astype(float)
    for i in range(len(df)):
        ts = df.iloc[i,:].to_numpy()
        ts = ts[~np.isnan(ts)]

        split_dict = train_test_split(ts, horizon, backhorizon, train_size, val_size, stride_length)
        for k,v in split_dict.items():
            train_test[k].append(v)

    for k,v in train_test.items():
        numpy_converted = np.array(v, dtype=object)
        if len(numpy_converted.shape) == 1:
            train_test[k] = numpy_converted
        else:
            train_test[k] = np.array(v)
    train_test["is_multivariate"] = True
    return train_test

# Note: All multivariate datasets have been formatted in the same way
#       Separate loading scripts here in case of individual differences

def read_exchange_rate(path):
    df = pd.read_csv(path).T
    df = df.iloc[1:,:]
    df.reset_index(inplace=True, drop=True)
    return df

def read_illness(path):
    df = pd.read_csv(path).T
    df = df.iloc[1:,:]
    df.reset_index(inplace=True, drop=True)
    return df

def read_ett(path):
    df = pd.read_csv(path).T
    df = df.iloc[1:,:]
    df.reset_index(inplace=True, drop=True)
    return df