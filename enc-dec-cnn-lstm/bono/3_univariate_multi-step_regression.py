import os
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow import keras


def exec(
    path='3_enc-dec-cnn-lstm_bono_univariate_multi-step_regression',
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    variant=1
):
    os.makedirs(path, exist_ok=True)

    # CODE
    file_name = '../../_data_/bono/bono-net.in_packets_sec.csv'
    timestamp_col = 'Timestamp'
    value_col = 'value'
    sep = ';'

    lag = 120
    split_ratio = 0.9  # 0.9 for training

    future_steps = 10

    # LOAD DATA

    # LOAD DATA
    data = pd.read_csv(file_name, sep=sep, parse_dates=[timestamp_col])
    data.index = data[timestamp_col]

    print(data.shape)
    print(data.head())

    data = data[10000:-20000]
    data = data.resample('15min').max()
    data = data.dropna()  # 20,83 dias

    print(data.shape)
    print(data.head())

    values = data[value_col].values

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
    x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

    print('Training Shape', x_tr.shape, y_tr.shape)
    print('Validation Shape', x_val.shape, y_val.shape)

    # (# of sequences, # of timesteps, length of features)
    print("Training shape (3 dimen)", x_tr.shape)

    ### MODEL ###

    # ### enc-dec-lstm ###
    # {'batch_size': 16, 'epochs': 200, 'hidden_layers': 5, 'learning_rate': 0.001, 'nodes_per_layer': 128, 'regularization': 0.2}

    n_outputs = y_tr.shape[1]
    model = keras.Sequential()

    if variant == 1:
        model.add(keras.layers.Conv1D(64, 3, padding='same',
                                      activation='tanh', input_shape=(x_tr.shape[1], x_tr.shape[2])))
        model.add(keras.layers.Conv1D(64, 5, padding='same',
                                      activation='tanh', input_shape=(x_tr.shape[1], x_tr.shape[2])))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.RepeatVector(n_outputs))
        model.add(keras.layers.LSTM(256, return_sequences=True))
        model.add(keras.layers.LSTM(256, return_sequences=True))
        model.add(keras.layers.LSTM(256, return_sequences=True))
        model.add(keras.layers.LSTM(256, return_sequences=True))
    if variant == 2:
        model.add(keras.layers.Conv1D(64, 3, padding='same',
                                      activation='tanh', input_shape=(x_tr.shape[1], x_tr.shape[2])))
        model.add(keras.layers.Conv1D(32, 5, padding='same',
                                      activation='tanh', input_shape=(x_tr.shape[1], x_tr.shape[2])))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.RepeatVector(n_outputs))
        model.add(keras.layers.LSTM(256, return_sequences=True))
        model.add(keras.layers.LSTM(256, return_sequences=True))
    if variant == 3:
        model.add(keras.layers.Conv1D(64, 3, padding='same',
                                      activation='tanh', input_shape=(x_tr.shape[1], x_tr.shape[2])))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.RepeatVector(n_outputs))
        model.add(keras.layers.LSTM(256, return_sequences=True))

    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Dense(32, activation='tanh')))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))

    model.summary()
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=learning_rate) if learning_rate != None else 'adam',
                  metrics=['mean_squared_error', 'mae', keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])

    # TRAIN
    tic = timeit.default_timer()

    history = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val), verbose=1)

    toc = timeit.default_timer()
    execution_time = toc - tic
    print(f"Executed in {str(execution_time)} seconds")

    model_name = 'model.h5'

    model.save(path + '/' + model_name)
    # model = keras.models.load_model(path + '/' + model_name)

    def visualize_loss(history, title):
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path + '/' + 'loss.png')
        # plt.show()

    if history != None:
        visualize_loss(history, "Training and Validation Loss")

    # model.load_weights('model.hdf5')

    ### PREDICT ###

    res = model.evaluate(x_val, y_val)
    print('res',  res)

    results = pd.DataFrame()
    results['Time To Train'] = [toc - tic]
    results['mse'] = [res[0]]
    results['mae'] = [res[2]]
    results['rmse'] = [res[3]]

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

        y_true = y_scaler.inverse_transform(y_true)
        y_pred = y_scaler.inverse_transform(
            np.array(y_pred).reshape(-1, future_steps))

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
        path=f'3_enc-dec-cnn-lstm_bono_univariate_multi-step_regression_{i}',
        epochs=200,
        batch_size=64,
        learning_rate=None,
        variant=1
    )
