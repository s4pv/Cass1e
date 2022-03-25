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

from datapreparation import Datapreparation
from modelplot import ModelPlot
from modelparameters import ModelParameters

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

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
            ds_f_c, ds_f_ohlv, ds_f_ohlcv = Datapreparation.Prepare_Data(dataset)
            # scaling data
            ohlcv_scaled = Datapreparation.Minmax_Scaler(ds_f_ohlcv, 'OHLCV_LSTM', coin)
            ohlv_scaled = Datapreparation.Minmax_Scaler(ds_f_ohlv, 'OHLV_LSTM', coin)
            c_scaled = Datapreparation.Minmax_Scaler(ds_f_c, 'C_LSTM', coin)
            #print(ohlcv_scaled.shape)
            #print(ohlv_scaled.shape)
            #print(c_scaled.shape)
            # Prediction Index
            train, test = Datapreparation.Dataset_Split(ohlcv_scaled)
            # for multivariate multi step predict purposes
            trainX, trainY = Datapreparation.Split_Sequences(train, N_STEPS_IN, N_STEPS_OUT)
            testX, testY = Datapreparation.Split_Sequences(test, N_STEPS_IN, N_STEPS_OUT)
            #print(trainX.shape, trainY.shape)
            #print(testX.shape, testY.shape)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            # Create and fit the Long short Term Memory network.
            lstm = Sequential()
            # so neurons are steps in * n features
            #n_neurons = trainX.shape[1] * trainX.shape[2]
            #print('Recommended neurons are')
            #print(n_neurons)
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
            #print(trainPredict.shape)
            testPredict = lstm.predict(testX, batch_size=N_STEPS_IN)
            #print(testPredict.shape)
            # evaluate loaded model
            lstm.evaluate(trainX, trainY, batch_size=N_STEPS_IN, verbose=VERBOSE)
            # reshape sets (only for not stacked LSTM)
            trainX = numpy.reshape(trainX, (-1, N_FEATURES))
            testX = numpy.reshape(testX, (-1, N_FEATURES))
            trainY = numpy.reshape(trainY, (-1, N_STEPS_OUT))
            testY = numpy.reshape(testY, (-1, N_STEPS_OUT))
            trainPredict = trainPredict[len(trainPredict)-1]
            testPredict = testPredict[len(trainPredict)-1]
            # invert sets
            trainPredict = Datapreparation.Invert_Transform(trainPredict, coin, 'C_LSTM', 'MINMAXSCALER')
            testPredict = Datapreparation.Invert_Transform(testPredict, coin, 'C_LSTM', 'MINMAXSCALER')
            trainX = Datapreparation.Invert_Transform(trainX, coin, 'OHLV_LSTM', 'MINMAXSCALER')
            testX = Datapreparation.Invert_Transform(testX, coin, 'OHLV_LSTM', 'MINMAXSCALER')
            trainY = Datapreparation.Invert_Transform(trainY, coin, 'C_LSTM', 'MINMAXSCALER')
            testY = Datapreparation.Invert_Transform(testY, coin, 'C_LSTM', 'MINMAXSCALER')
            ds_f_ohlcv = Datapreparation.Invert_Transform(ohlcv_scaled, coin, 'OHLCV_LSTM', 'MINMAXSCALER')
            ds_f_ohlv = Datapreparation.Invert_Transform(ohlv_scaled, coin, 'OHLV_LSTM', 'MINMAXSCALER')
            ds_f_c = Datapreparation.Invert_Transform(c_scaled, coin, 'C_LSTM', 'MINMAXSCALER')
            # printing shapes
            #print(trainPredict.shape)
            #print(testPredict.shape)
            #print(trainX.shape)
            #print(testX.shape)
            #print(trainY.shape)
            #print(testY.shape)
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
            ModelPlot.Plot_Actual(ds_f_c, trainYPlot, testYPlot, trainPredictPlot, testPredictPlot, coin, 'LSTM2')
            # Saving model to disk
            ModelParameters.Save_Model(lstm, coin, 'LSTM')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True
