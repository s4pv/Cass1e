from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional

import math
import warnings
from sklearn.metrics import mean_squared_error
import numpy
import os
from mlautomator.mlautomator import MLAutomator

from helper import Helper
from datapreparation import Datapreparation
from modelplot import ModelPlot
from modelparameters import ModelParameters
from stats import Stats
import pandas as pd

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
ALGO_LIST = parsed_config['ml_options']['ALGO_LIST']

ML_MODEL = parsed_config['model_options']['ML_MODEL']
WEIGHT = parsed_config['model_options']['WEIGHT']

class MachineLearning:
    def Model(dataset, date, coin):
        try:
            print('Transforming the data to returns to make the data stationary')
            #return_dataset = Datapreparation.Returns_Transformation(dataset)
            return_dataset = Datapreparation.Fractional_Differentiation(dataset, WEIGHT)
            aux_c, aux_ohlv, aux_ohlcv = Datapreparation.Prepare_Data(dataset)
            aux_train, aux_test = Datapreparation.Dataset_Split(aux_ohlcv)
            a = 'Testing if the data is stationary on close data'
            print(a)
            statsDF, pDF = Stats.Dickey_Fuller(return_dataset['close'], coin, 'model', date)
            # preprocessing data
            r_c, r_ohlv, r_ohlcv = Datapreparation.Prepare_Data(return_dataset)
            # scaling data
            print('Starting to normalize the set with the quantile gaussian transformation (robust to outliers)')
            r_ohlcv_scaled = Datapreparation.Gaussian_Scaler(r_ohlcv, 'OHLCV_LSTM', coin, date)
            #r_ohlcv_scaled = Datapreparation.PowerTransformer(r_ohlcv, 'OHLV_LSTM', coin, date)
            r_ohlv_scaled = Datapreparation.Gaussian_Scaler(r_ohlv, 'OHLV_LSTM', coin, date)
            #r_ohlv_scaled = Datapreparation.PowerTransformer(r_ohlv, 'OHLV_LSTM', coin, date)
            r_c_scaled = Datapreparation.Gaussian_Scaler(r_c, 'C_LSTM', coin, date)
            #r_c_scaled = Datapreparation.PowerTransformer(r_c, 'C_LSTM', coin, date)
            #print(ohlcv_scaled.shape)
            #print(ohlv_scaled.shape)
            #print(c_scaled.shape)
            # reshaping data
            r_c_scaled = pd.DataFrame(r_c_scaled)
            r_c_scaled.columns = ['close']
            # plot histogram probplot and qplot
            print('Starting to plot histogram, PPplot and QQplot on close data')
            Stats.Plots(r_c_scaled['close'], coin, 'model', date)
            # testing for normality
            b = 'Testing for normality on close data sets.'
            print(b)
            statsSWC, pSWC = Stats.Shapiro_Wilk(r_c_scaled['close'], coin, 'model', date)
            # Split sequences
            train, test = Datapreparation.Dataset_Split(r_ohlcv_scaled)
            # train and test split LSTM
            trainX, trainY = Datapreparation.Split_Sequences(train, N_STEPS_IN, N_STEPS_OUT)
            testX, testY = Datapreparation.Split_Sequences(test, N_STEPS_IN, N_STEPS_OUT)
            # train and test split for ML-Automator
            #trainX, trainY = Datapreparation.Split_Sequences(train, N_STEPS_IN, 1)
            #testX, testY = Datapreparation.Split_Sequences(test, N_STEPS_IN, 1)
            #trainX = numpy.reshape(trainX, (-1, N_FEATURES))
            #testX = numpy.reshape(testX, (-1, N_FEATURES))
            #trainY = trainY[len(trainY)-1].reshape(N_STEPS_OUT, -1)
            #testY = testY[len(testY)-1].reshape(N_STEPS_OUT, -1)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            # Create and fit the Long short Term Memory network.
            model = Sequential()
            # so neurons are steps in * n features
            #n_neurons = trainX.shape[1] * trainX.shape[2]
            #print('Recommended neurons are')
            #print(n_neurons)
            # Multi variable LSTM multi step output
            model.add(LSTM(NEURONS, input_shape=(N_STEPS_IN, N_FEATURES), return_sequences=True))
            # Stacked multi variable LSTM bidirectional multi step output
            #model.add(Bidirectional(LSTM(NEURONS, batch_input_shape=(N_STEPS_IN, N_STEPS_OUT, N_FEATURES), stateful=True, return_sequences=True)))
            #model.add(Bidirectional(LSTM(NEURONS, batch_input_shape=(N_STEPS_IN, N_STEPS_OUT, N_FEATURES), stateful=True)))

            #model = MLAutomator(trainX, trainY, iterations=30, algo_type='regressor', score_metric='neg_mean_squared_error',
            #                        specific_algos=ALGO_LIST)
            #model.find_best_algorithm()
            #model.fit_best_pipeline()
            #model.print_best_space()
            #filedir = 'model_parameters/' + str(date) + '/'
            #os.makedirs(filedir, exist_ok=True)
            #model.save_best_pipeline(filedir)

            model.add(Dense(N_STEPS_OUT))
            model.add(Dropout(DROPOUT))
            model.compile(loss='mean_squared_error', optimizer='adam')
            # no memory between batches
            model.fit(trainX, trainY, epochs=EPOCH, batch_size=N_STEPS_IN, verbose=VERBOSE)
            model.reset_states()
            # memory between batches
            #for i in range(EPOCH):
            #    model.fit(trainX, trainY, epochs=1, batch_size=N_STEPS_IN, verbose=VERBOSE, shuffle=False)
            #    model.reset_states()
            #print summary
            print(model.summary())
            # make predictions LSTM
            trainPredict = model.predict(trainX, batch_size=N_STEPS_IN)
            model.reset_states()
            #print(trainPredict.shape)
            testPredict = model.predict(testX, batch_size=N_STEPS_IN)
            model.reset_states()
            #print(testPredict.shape)
            # make preditions ML-Automator
            #trainPredict = model.predict(trainX)
            #testPredict = model.predict(testX)
            # evaluate loaded model
            #model.evaluate(trainX, trainY, batch_size=N_STEPS_IN, verbose=VERBOSE)
            # reshape sets
            trainPredict = trainPredict[len(trainPredict)-1].reshape(N_STEPS_OUT, -1)
            testPredict = testPredict[len(testPredict)-1].reshape(N_STEPS_OUT, -1)
            trainX = numpy.reshape(trainX, (-1, N_FEATURES))
            testX = numpy.reshape(testX, (-1, N_FEATURES))
            trainY = trainY[len(trainY)-1].reshape(N_STEPS_OUT, -1)
            testY = testY[len(testY)-1].reshape(N_STEPS_OUT, -1)
            # invert sets
            trainPredict_unscaled = Datapreparation.Invert_Transform(trainPredict, coin, 'C_LSTM', 'GAUSSIANSCALER')
            testPredict_unscaled = Datapreparation.Invert_Transform(testPredict, coin, 'C_LSTM', 'GAUSSIANSCALER')
            trainX_unscaled = Datapreparation.Invert_Transform(trainX, coin, 'OHLV_LSTM', 'GAUSSIANSCALER')
            testX_unscaled = Datapreparation.Invert_Transform(testX, coin, 'OHLV_LSTM', 'GAUSSIANSCALER')
            trainY_unscaled = Datapreparation.Invert_Transform(trainY, coin, 'C_LSTM', 'GAUSSIANSCALER')
            testY_unscaled = Datapreparation.Invert_Transform(testY, coin, 'C_LSTM', 'GAUSSIANSCALER')
            r_ohlcv_unscaled = Datapreparation.Invert_Transform(r_ohlcv_scaled, coin, 'OHLCV_LSTM', 'GAUSSIANSCALER')
            r_ohlv_unscaled = Datapreparation.Invert_Transform(r_ohlv_scaled, coin, 'OHLV_LSTM', 'GAUSSIANSCALER')
            r_c_unscaled = Datapreparation.Invert_Transform(r_c_scaled, coin, 'C_LSTM', 'GAUSSIANSCALER')
            # printing shapes
            #print(trainPredict_unscaled.shape)
            #print(testPredict_unscaled.shape)
            #print(trainX_unscaled.shape)
            #print(testX_unscaled.shape)
            #print(trainY_unscaled.shape)
            #print(testY_unscaled.shape)
            #print(r_ohlcv_unscaled.shape)
            #print(r_ohlv_unscaled.shape)
            #print(r_c_unscaled.shape)
            # convert back from returns to price
            #p_trainPredict_u = Datapreparation.Price_Transformation(trainPredict_unscaled, aux_train, 'model')
            #p_testPredict_u = Datapreparation.Price_Transformation(testPredict_unscaled, aux_test, 'model')
            #p_trainX_u = Datapreparation.Price_Transformation(trainX_unscaled, dataset, 'model')
            #p_testX_u = Datapreparation.Price_Transformation(testX_unscaled, dataset, 'model')
            #p_trainY_u = Datapreparation.Price_Transformation(trainY_unscaled, aux_train, 'model')
            #p_testY_u = Datapreparation.Price_Transformation(testY_unscaled, aux_test, 'model')
            #p_ohlcv_u = Datapreparation.Price_Transformation(r_ohlcv_unscaled, dataset, 'model')
            #p_ohlv_u = Datapreparation.Price_Transformation(r_ohlv_unscaled, dataset, 'model')
            #p_c_u = Datapreparation.Price_Transformation(r_c_unscaled, dataset, 'model')
            # convert back from fractional differentiation to price
            p_trainPredict_u = Datapreparation.Fractional_Integration(trainPredict_unscaled, aux_train, WEIGHT, 'model')
            p_testPredict_u = Datapreparation.Fractional_Integration(testPredict_unscaled, aux_test, WEIGHT, 'model')
            #p_trainX_u = Datapreparation.Fractional_Integration(trainX_unscaled, dataset, WEIGHT, 'model')
            #p_testX_u = Datapreparation.Fractional_Integration(testX_unscaled, dataset, WEIGHT, 'model')
            p_trainY_u = Datapreparation.Fractional_Integration(trainY_unscaled, aux_train, WEIGHT, 'model')
            p_testY_u = Datapreparation.Fractional_Integration(testY_unscaled, aux_test, WEIGHT, 'model')
            #p_ohlcv_u = Datapreparation.Fractional_Integration(r_ohlcv_unscaled, dataset, WEIGHT, 'model')
            #p_ohlv_u = Datapreparation.Fractional_Integration(r_ohlv_unscaled, dataset, WEIGHT, 'model')
            p_c_u = Datapreparation.Fractional_Integration(r_c_unscaled, dataset, WEIGHT, 'model')
            #reshape data for comparing agains correct close data
            #print(p_trainPredict_u.shape)
            #print(p_testPredict_u.shape)
            #print(p_trainX_u.shape)
            #print(p_testX_u.shape)
            #print(p_trainY_u.shape)
            #print(p_testY_u.shape)
            #print(p_ohlcv_u.shape)
            #print(p_ohlv_u.shape)
            #print(p_c_u.shape)
            # Train Metrics
            trainScore = math.sqrt(mean_squared_error(p_trainY_u, p_trainPredict_u))
            c = 'Train Score: %.2f RMSE' % (trainScore)
            print(c)
            testScore = math.sqrt(mean_squared_error(p_testY_u, p_testPredict_u))
            d = 'Test Score: %.2f RMSE' % (testScore)
            print(d)
            # Estimate model performance
            #aux for stats
            aux_testY = numpy.squeeze(numpy.asarray(p_testY_u))
            aux_testPredict = numpy.squeeze(numpy.asarray(p_testPredict_u))
            # Spearman (R2) just for parametric/normally distributed data
            #stat1, p1 = Stats.Spearman(aux_testY, aux_testPredict)
            # Pearson test (R2) for non-parametric/not normally distributed data
            stat2, p2 = Stats.Pearson(aux_testY, aux_testPredict, coin, 'model', date)
            # Kendall test
            #stat3, p3 = Stats.Kendall(aux_testY, aux_testPredict)
            # Chi Squared test (Xi2) for non-parametric/not normally distributed data
            stat4, p4 = Stats.Chi_Squared(aux_testY, aux_testPredict, coin, 'model', date)
            # Mean Absolute Error (MAE)
            MAE = mean_squared_error(p_testY_u, p_testPredict_u)
            e = f'Median Absolute Error (MAE): {numpy.round(MAE, 2)}'
            print(e)
            # Mean Absolute Percentage Error (MAPE)
            MAPE = numpy.mean((numpy.abs(numpy.subtract(p_testY_u, p_testPredict_u) / p_testY_u))) * 100
            f = f'Mean Absolute Percentage Error (MAPE): {numpy.round(MAPE[0], 2)} %'
            print(f)
            # Median Absolute Percentage Error (MDAPE)
            MDAPE = numpy.median((numpy.abs(numpy.subtract(p_testY_u, p_testPredict_u) / p_testY_u))) * 100
            g = f'Median Absolute Percentage Error (MDAPE): {numpy.round(MDAPE, 2)} %'
            print(g)
            # saving model fit stats
            lines = [str(c), str(d), str(e), str(f), str(g)]
            filedir = 'model_stats/' + str(date) + '/'
            filename = os.path.join(filedir, str(coin['symbol']) + '_model_fit_stats.txt')
            os.makedirs(filedir, exist_ok=True)
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
            # reshape dataset to plot correctly close data
            p_c_unscaled_2 = dataset['close'].to_numpy()
            p_c_unscaled_2 = numpy.reshape(p_c_unscaled_2, (len(p_c_unscaled_2), 1))
            p_c_unscaled_2 = numpy.delete(p_c_unscaled_2, 0, 0)
            p_c_unscaled_2 = pd.DataFrame(p_c_unscaled_2)
            p_c_unscaled_2 = Datapreparation.Reshape_Float(p_c_unscaled_2)
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot, closePlot1 = ModelPlot.Shift_Plot(p_c_unscaled_2, len(trainX_unscaled), p_trainPredict_u, p_testPredict_u)
            trainYPlot, testYPlot, closePlot2 = ModelPlot.Shift_Plot(p_c_unscaled_2, len(trainX_unscaled), p_trainY_u, p_testY_u)
            # plot baseline and predictions
            closePlot1 = numpy.reshape(closePlot1, (1, len(closePlot1)))
            ModelPlot.Plot_Actual(closePlot1[0], trainYPlot, testYPlot, trainPredictPlot, testPredictPlot, coin, ML_MODEL, date)
            # Saving model to disk
            ModelParameters.Save_Model(model, coin, ML_MODEL, date)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True
