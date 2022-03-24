from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional

import math
import warnings
from sklearn.metrics import mean_squared_error
from helper import Helper

import numpy

from preprocessing import Preprocessing
from modelplot import ModelPlot
from modelparameters import ModelParameters

import pandas as pd

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

NO_DAYS = parsed_config['model_options']['NO_DAYS']

TRAIN_SIZE = parsed_config['ml_options']['TRAIN_SIZE']
NEURONS = parsed_config['ml_options']['NEURONS']
DROPOUT = parsed_config['ml_options']['DROPOUT']
N_STEPS_IN = parsed_config['ml_options']['N_STEPS_IN']
N_STEPS_OUT = parsed_config['ml_options']['N_STEPS_OUT']
EPOCH = parsed_config['ml_options']['EPOCH']
N_FEATURES = parsed_config['ml_options']['N_FEATURES']
VERBOSE = parsed_config['ml_options']['VERBOSE']


class MachineLearning:
    def LSTM(dataset, coin):
        try:
            # preprocessing data
            ds_f_c, ds_f_ohlv, ds_f_ohlcv = Preprocessing.Prepare_Data(dataset)
            # scaling data
            ohlcv_scaled = Preprocessing.Minmax_Scaler(ds_f_ohlcv, 'OHLCV_LSTM', coin)
            ohlv_scaled = Preprocessing.Minmax_Scaler(ds_f_ohlv, 'OHLV_LSTM', coin)
            c_scaled = Preprocessing.Minmax_Scaler(ds_f_c, 'C_LSTM', coin)
            print(ohlcv_scaled.shape)
            print(ohlv_scaled.shape)
            print(c_scaled.shape)
            # Prediction Index
            train, test = Preprocessing.Dataset_Split(ohlcv_scaled)
            # for multivariate multi step predict purposes
            trainX, trainY = Preprocessing.Split_Sequences(train, N_STEPS_IN, N_STEPS_OUT)
            testX, testY = Preprocessing.Split_Sequences(test, N_STEPS_IN, N_STEPS_OUT)
            print(trainX.shape, trainY.shape)
            print(testX.shape, testY.shape)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            # Create and fit the Long short Term Memory network.
            lstm = Sequential()
            # so neurons are steps in * n features
            n_neurons = trainX.shape[1] * trainX.shape[2]
            print('Recommended neurons are')
            print(n_neurons)
            # not stacked multi variable LSTM bidirectional multi step output
            lstm.add(Bidirectional(LSTM(NEURONS, input_shape=(N_STEPS_IN, N_FEATURES), return_sequences=True)))
            # Stacked multi variable LSTM bidirectional multi step output
            #lstm.add(Bidirectional(LSTM(n_neurons, batch_input_shape=(N_STEPS_IN, N_STEPS_OUT, N_FEATURES), stateful=True, return_sequences=True)))
            #lstm.add(Bidirectional(LSTM(n_neurons, batch_input_shape=(N_STEPS_IN, N_STEPS_OUT, N_FEATURES), stateful=True)))
            lstm.add(Dense(N_STEPS_OUT))
            lstm.add(Dropout(DROPOUT))
            lstm.compile(loss='mean_squared_error', optimizer='adam')
            # no memory between batches
            lstm.fit(trainX, trainY, epochs=EPOCH, batch_size=N_STEPS_IN, verbose=VERBOSE)
            lstm.reset_states()
            # memory between batches
            #for i in range(EPOCH):
            #    lstm.fit(trainX, trainY, epochs=1, batch_size=N_STEPS_IN, verbose=VERBOSE, shuffle=False)
            #    lstm.reset_states()
            print(lstm.summary())
            # make predictions
            trainPredict = lstm.predict(trainX, batch_size=N_STEPS_IN)
            lstm.reset_states()
            print(trainPredict.shape)
            testPredict = lstm.predict(testX, batch_size=N_STEPS_IN)
            print(testPredict.shape)
            # reshape sets (only for not stacked LSTM)
            trainX = numpy.reshape(trainX, (-1, N_FEATURES))
            testX = numpy.reshape(testX, (-1, N_FEATURES))
            trainY = numpy.reshape(trainY, (-1, N_STEPS_OUT))
            testY = numpy.reshape(testY, (-1, N_STEPS_OUT))
            trainPredict = trainPredict[len(trainPredict)-1]
            testPredict = testPredict[len(trainPredict)-1]
            # invert sets
            trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'C_LSTM', 'MINMAXSCALER')
            testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'C_LSTM', 'MINMAXSCALER')
            trainX = Preprocessing.Invert_Transform(trainX, coin, 'OHLV_LSTM', 'MINMAXSCALER')
            testX = Preprocessing.Invert_Transform(testX, coin, 'OHLV_LSTM', 'MINMAXSCALER')
            trainY = Preprocessing.Invert_Transform(trainY, coin, 'C_LSTM', 'MINMAXSCALER')
            testY = Preprocessing.Invert_Transform(testY, coin, 'C_LSTM', 'MINMAXSCALER')
            ds_f_ohlcv = Preprocessing.Invert_Transform(ohlcv_scaled, coin, 'OHLCV_LSTM', 'MINMAXSCALER')
            ds_f_ohlv = Preprocessing.Invert_Transform(ohlv_scaled, coin, 'OHLV_LSTM', 'MINMAXSCALER')
            ds_f_c = Preprocessing.Invert_Transform(c_scaled, coin, 'C_LSTM', 'MINMAXSCALER')
            # printing shapes
            print(trainPredict.shape)
            print(testPredict.shape)
            print(trainX.shape)
            print(testX.shape)
            print(trainY.shape)
            print(testY.shape)
            # reshape for not staked
            testPredict.reshape(-1, N_STEPS_OUT)
            trainPredict.reshape(-1, N_STEPS_OUT)
            testY = testY[len(testY)-1].reshape(-1, N_STEPS_OUT)
            trainY = trainY[len(trainY)-1].reshape(-1, N_STEPS_OUT)
            # Estimate model performance
            # Mean Absolute Error (MAE)
            MAE = mean_squared_error(testY, testPredict)
            print(f'Median Absolute Error (MAE): {numpy.round(MAE, 2)}')
            # Mean Absolute Percentage Error (MAPE)
            MAPE = numpy.mean((numpy.abs(numpy.subtract(testY, testPredict) / testY))) * 100
            print(f'Mean Absolute Percentage Error (MAPE): {numpy.round(MAPE, 2)} %')
            # Median Absolute Percentage Error (MDAPE)
            MDAPE = numpy.median((numpy.abs(numpy.subtract(testY, testPredict) / testY))) * 100
            print(f'Median Absolute Percentage Error (MDAPE): {numpy.round(MDAPE, 2)} %')
            trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY, testPredict))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds_f_c, len(trainX), trainPredict, testPredict)
            trainYPlot, testYPlot = ModelPlot.Shift_Plot(ds_f_c, len(trainX), trainY, testY)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds_f_c, trainYPlot, testYPlot, trainPredictPlot, testPredictPlot, coin, 'LSTM')
            # Saving model to disk
            ModelParameters.Save_Model(lstm, coin, 'LSTM')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True
