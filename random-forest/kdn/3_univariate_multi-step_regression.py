import os
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import joblib


def exec(
    path='3_random-forest_kdn_univariate_multi-step_regression',
    bootstrap=True,
    n_estimators=200,
    max_features='auto',
    max_depth=30,
    random_state=420,
):
    os.makedirs(path, exist_ok=True)

    # CODE
    file_name = '../../_data_/kdn/cpu.csv'
    value_col = 'Cpu'
    sep = ';'

    lag = 60  # 168 hours = 1 week
    split_ratio = 0.9  # 0.9 for training

    future_steps = 10

    # LOAD DATA
    data = pd.read_csv(file_name, sep=sep)

    data['ind'] = [i for i in range(0, len(data[value_col]))]
    print(data)

    data = data[['ind', value_col]]
    data.index = data['ind']
    print(data.shape)
    print(data.head())

    # homestead-cpu.idle_perc.csv
    # data[value_col] = 100 - data[value_col]

    # data = data.resample('5T').mean()
    # data = data.fillna(data.interpolate())

    print('Data Shape', data.shape)  # 177097

    # data = data[10000:100000]

    # others
    values = data[value_col].values

    ### PLOT DATA ###
    # ar = np.arange(len(values))
    # plt.figure(figsize=(22, 5))
    # plt.plot(ar, values, 'r')
    # plt.legend(["# of values"])
    # plt.show()

    # sample = values[:168]
    # ar = np.arange(len(sample))
    # plt.figure(figsize=(22, 5))
    # plt.plot(ar, sample, 'r')
    # # plt.show()

    ### PRE PROCESSING ###

    def split_data(seq, num, future):
        x, y = [], []
        print('Received SEQ', seq)
        # remove num value for complete sequences
        for i in range(0, (len(seq)-(num + future)), 1):
            input_ = seq[i:i+num]
            output = seq[i+num:i+num+future]
            x.append(input_)
            y.append(output)

        return np.array(x), np.array(y)

    x, y = split_data(values, lag, future_steps)
    print('Length of X', len(x), 'each are some array with', len(x[0]))
    print('Length of Y', len(y), 'value', y[0])

    # SPLIT THE DATASET (training and validation sets)
    ind = int(split_ratio * len(x))

    x_tr = x[:ind]
    y_tr = y[:ind]

    x_val = x[ind:]
    y_val = y[ind:]

    print('Length of Training Set', len(x_tr))
    print('Length of Validation Set', len(x_val))

    # NORMALIZE
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    x_tr = x_scaler.fit_transform(x_tr)
    x_val = x_scaler.transform(x_val)
    # reshaping the output for normalization (add each value on array)
    # y_tr = y_tr.reshape(len(y_tr), 1)
    # y_val = y_val.reshape(len(y_val), 1)
    # normalize the output (and remove each value from array)
    y_tr = y_scaler.fit_transform(y_tr)
    y_val = y_scaler.transform(y_val)

    # RESHAPING THE DATA TO 3 DIMENSIONS
    # reshaping input data (reshape from 2 dimensions to 3 dimensions)
    # third dimension is the length of the vectors of the sequence elements.
    # x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1)
    # x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

    print('Training Shape', x_tr.shape, y_tr.shape)
    print('Validation Shape', x_val.shape, y_val.shape)

    # (# of sequences, # of timesteps, length of features)
    print("Training shape (3 dimen)", x_tr.shape)

    ### MODEL ###

    # ### GRU ###
    # {'batch_size': 16, 'epochs': 200, 'hidden_layers': 5, 'learning_rate': 0.001, 'nodes_per_layer': 128, 'regularization': 0.2}

    tic = timeit.default_timer()

    model = MultiOutputRegressor(
        RandomForestRegressor(
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            random_state=random_state,
        ))

    model.fit(x_tr, y_tr)

    toc = timeit.default_timer()
    execution_time = toc - tic
    print(f"Executed in {str(execution_time)} seconds")

    model_name = 'model.joblib'

    joblib.dump(model, model_name)
    # model = joblib.load(model_name)

    ### PREDICT ###

    predicted = model.predict(x_val)

    print('predicted', predicted)

    mse = mean_squared_error(y_val, predicted)
    mae = mean_absolute_error(y_val, predicted)
    rmse = mean_squared_error(y_val, predicted, squared=False)
    print('MSE | MAE | RMSE',  mse, mae, rmse)

    results = pd.DataFrame()
    results['Time To Train'] = [toc - tic]
    results['mse'] = [mse]
    results['mae'] = [mae]
    results['rmse'] = [rmse]

    results.to_csv(path + '/' + 'results.csv', index=False)

    def plotRealVsForecasted(y_true, y_pred):
        ar = np.arange(len(y_true))
        plt.figure(figsize=(22, 5))
        plt.plot(ar, y_true, 'r', label="Real Values")
        plt.plot(ar, y_pred, 'y', label="Predicted Values")
        plt.legend()
        plt.savefig(path + '/' + 'real_fore.png')
        # plt.show()

    def compare():
        y_pred = []
        times = []

        print(x_tr.shape)
        print(x_val.shape)

        for item in x_val:
            tic = timeit.default_timer()
            y_pred.extend(model.predict(np.array([item]))[0])
            toc = timeit.default_timer()
            times.append(toc - tic)

        y_true = y_val

        # y_true = y_scaler.inverse_transform(
        #     np.array(y_true).reshape(y_true.shape[0] * y_true.shape[1]).reshape(-1, 1))
        # y_pred = y_scaler.inverse_transform(
        #     np.array(y_pred).reshape(y_pred.shape[0] * y_pred.shape[1], 1))

        y_true = y_scaler.inverse_transform(y_true)
        y_pred = y_scaler.inverse_transform(
            np.array(y_pred).reshape(-1, future_steps))

        print('y_true.shape', y_true.shape)
        print('y_pred.shape', y_pred.shape)

        compare = pd.DataFrame()
        compare['results'] = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1])
        compare.to_csv(path + '/' + 'predictions.csv', index=False)

        compare = pd.DataFrame()
        compare['times'] = times
        compare.to_csv(path + '/' + 'time_to_predict.csv', index=False)

        plotRealVsForecasted(y_true.reshape(y_true.shape[0] * y_true.shape[1]).reshape(
            -1, 1), y_pred.reshape(y_pred.shape[0] * y_pred.shape[1]).reshape(-1, 1))

    compare()


for i in range(1, 11):
    exec(
        path=f'3_random-forest_kdn_univariate_multi-step_regression_{i}',
        bootstrap=True,
        n_estimators=1000,
        max_features='log2',
        max_depth=40,
        random_state=420,
    )