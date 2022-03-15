import math
import warnings
from sklearn.metrics import mean_squared_error
from helper import Helper
from modelplot import ModelPlot
from modelparameters import ModelParameters

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model

import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import Preprocessing
import numpy

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

TO_FORECAST = parsed_config['forecast_options']['TO_FORECAST']

class Finance:

    # Finance.ARIMA(df, 0,0,1) # MA model
    # Finance.ARIMA(df, 2,0,1) # ARMA model
    # Finance.ARIMA(df, 1, 1, 1) # ARIMA model

    def AR(dataset, lag, coin):
        try:
            ds = Preprocessing.Reshape_Float(dataset)
            ds = Preprocessing.Minmax_Scaler(ds, 'AR', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            print('Starting to fit model: AR')
            # fit model
            model = AutoReg(trainX, lags=lag)
            model_fit = model.fit()
            # make prediction
            #testX = model_fit.predict(len(trainX), len(trainX))
            #print('Predicted next value of: ')
            #print(testX)
            model_fit.plot_diagnostics(figsize=(10, 6))
            plt.show()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def ARIMA(dataset, order1, order2, order3, coin):
        try:
            ds = Preprocessing.Reshape_Float(dataset)
            ds = Preprocessing.Minmax_Scaler(ds, 'ARIMA', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            print('Starting to fit model: ARIMA')
            # fit model
            model = ARIMA(trainX, order=(order1, order2, order3))
            model_fit = model.fit()
            # make prediction
            #testX = model_fit.predict(len(trainX), len(trainX), typ='levels')
            #print('Predicted next value of: ')
            #print(testX)
            model_fit.plot_diagnostics(figsize=(10, 6))
            plt.show()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def SARIMA(dataset, order1, order2, order3, sorder1, sorder2, sorder3, sorder4, predict, coin):
        try:
            ds = Preprocessing.Reshape_Float(dataset)
            ds = Preprocessing.Minmax_Scaler(ds, 'SARIMA', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            print('Starting to fit model: SARIMA')
            # fit model
            model = SARIMAX(trainX, order=(order1, order2, order3),
                            seasonal_order=(sorder1, sorder2, sorder3, sorder4),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit()
            # make prediction
            #testX = model_fit.predict(len(trainX), len(trainX))
            #print('Predicted next value of: ')
            #print(testX)
            model_fit.plot_diagnostics(figsize=(10, 6))
            plt.show()
            print(model_fit.summary())
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def SARIMAX(dataset1, dataset2, dataset3, order1, order2, order3, sorder1, sorder2, sorder3, sorder4, predict,
                coin):
        try:
            ds1 = Preprocessing.Reshape_Float(dataset1)
            ds2 = Preprocessing.Reshape_Float(dataset2)
            ds3 = Preprocessing.Reshape_Float(dataset3)
            ds1 = Preprocessing.Minmax_Scaler(ds1, 'SARIMAX', coin)
            ds2 = Preprocessing.Minmax_Scaler(ds2, 'SARIMAX', coin)
            ds3 = Preprocessing.Minmax_Scaler(ds3, 'SARIMAX', coin)
            train1, test1 = Preprocessing.Dataset_Split(ds1)
            train2, test2 = Preprocessing.Dataset_Split(ds2)
            train3, test3 = Preprocessing.Dataset_Split(ds3)
            trainX1, trainY1, testX1, testY1 = Preprocessing.Reshape_Data(train1, test1)
            trainX2, trainY2, testX2, testY2 = Preprocessing.Reshape_Data(train2, test2)
            trainX3, trainY3, testX3, testY3 = Preprocessing.Reshape_Data(train3, test3)
            print('Starting to fit model: SARIMAX')
            # fit model
            model = SARIMAX(trainX1, trainX2, order=(order1, order2, order3), seasonal_order=(sorder1, sorder2,
                                                                                              sorder3, sorder4))
            model_fit = model.fit(disp=False)
            # make prediction
            #exog2 = [len(trainX3) + predict]
            #testX1 = model_fit.predict(len(trainX1), len(trainX1), exog=[exog2])
            #print('Predicted next value of: ')
            #print(testX1)
            model_fit.plot_diagnostics(figsize=(10, 6))
            plt.show()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def VAR(dataset1, dataset2, coin):
        try:
            ds1 = Preprocessing.Reshape_Float(dataset1)
            ds2 = Preprocessing.Reshape_Float(dataset2)
            ds1 = Preprocessing.Minmax_Scaler(ds1, 'VAR', coin)
            ds2 = Preprocessing.Minmax_Scaler(ds2, 'VAR', coin)
            train1, test1 = Preprocessing.Dataset_Split(ds1)
            train2, test2 = Preprocessing.Dataset_Split(ds2)
            trainX1, trainY1, testX1, testY1 = Preprocessing.Reshape_Data(train1, test1)
            trainX2, trainY2, testX2, testY2 = Preprocessing.Reshape_Data(train2, test2)
            data = list()
            for i in range(len(trainX1)):
                v1 = trainX1[i]
                v2 = trainX2[i]
                row = [v1, v2]
                data.append(row)
            print('Starting to fit model: VAR')
            # fit model
            model = VAR(data)
            model_fit = model.fit()
            # make prediction
            #testX1 = model_fit.forecast(model.y, steps=1)
            #print('Predicted next value of: ')
            #print(testX1)
            print(model_fit.summary())
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def VARMA(dataset1, dataset2, order1, order2, coin):
        try:
            ds1 = Preprocessing.Reshape_Float(dataset1)
            ds2 = Preprocessing.Reshape_Float(dataset2)
            ds1 = Preprocessing.Minmax_Scaler(ds1, 'VARMA', coin)
            ds2 = Preprocessing.Minmax_Scaler(ds2, 'VARMA', coin)
            train1, test1 = Preprocessing.Dataset_Split(ds1)
            train2, test2 = Preprocessing.Dataset_Split(ds2)
            trainX1, trainY1, testX1, testY1 = Preprocessing.Reshape_Data(train1, test1)
            trainX2, trainY2, testX2, testY2 = Preprocessing.Reshape_Data(train2, test2)
            data = list()
            for i in range(len(trainX1)):
                v1 = trainX1[i]
                v2 = trainX2[i]
                row = [v1, v2]
                data.append(row)
            print('Starting to fit model: VARMA')
            # fit model
            model = VARMAX(data, order=(order1, order2))
            model_fit = model.fit(disp=False)
            # make prediction
            #testX1 = model_fit.forecast()
            #print('Predicted next value of: ')
            #print(testX1)
            print(model_fit.summary())
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def VARMAX(dataset1, dataset2, dataset3, order1, order2, coin):
        try:
            ds1 = Preprocessing.Reshape_Float(dataset1)
            ds2 = Preprocessing.Reshape_Float(dataset2)
            ds3 = Preprocessing.Reshape_Float(dataset3)
            ds1 = Preprocessing.Minmax_Scaler(ds1, 'VARMAX', coin)
            ds2 = Preprocessing.Minmax_Scaler(ds2, 'VARMAX', coin)
            ds3 = Preprocessing.Minmax_Scaler(ds3, 'VARMAX', coin)
            train1, test1 = Preprocessing.Dataset_Split(ds1)
            train2, test2 = Preprocessing.Dataset_Split(ds2)
            train3, test3 = Preprocessing.Dataset_Split(ds3)
            trainX1, trainY1, testX1, testY1 = Preprocessing.Reshape_Data(train1, test1)
            trainX2, trainY2, testX2, testY2 = Preprocessing.Reshape_Data(train2, test2)
            trainX3, trainY3, testX3, testY3 = Preprocessing.Reshape_Data(train3, test3)
            data = list()
            data_exog = list()
            for i in range(len(trainX1)):
                v1 = trainX1[i]
                v2 = trainX2[i]
                v3 = trainX3[i]
                row1 = [v1, v2]
                row2 = [v3]
                data.append(row1)
                data_exog.append(row2)
            print('Starting to fit model: VARMAX')
            # fit model
            model = VARMAX(data, exog=data_exog, order=(order1, order2))
            model_fit = model.fit(disp=False)
            # make prediction
            data_exog2 = [[len(data_exog)]]
            #testX1 = model_fit.forecast(exog=data_exog2)
            #print('Predicted next value of: ')
            #print(testX1)
            print(model_fit.summary())
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def SES(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Float(dataset)
            ds = Preprocessing.Minmax_Scaler(ds, 'SES', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            data = list()
            for i in range(len(trainX)):
                row = trainX[i]
                data.append(row)
            print('Starting to fit model: SES')
            # fit model
            model = SimpleExpSmoothing(data)
            model_fit = model.fit()
            # make prediction
            #testX = model_fit.predict(len(data), len(data))
            #print('Predicted next value of: ')
            #print(testX)
            print(model_fit.summary())
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def HWES(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Float(dataset)
            ds = Preprocessing.Minmax_Scaler(ds, 'HWES', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            data = list()
            for i in range(len(trainX)):
                row = trainX[i]
                data.append(row)
            print('Starting to fit model: HWES')
            # fit model
            model = ExponentialSmoothing(data)
            model_fit = model.fit()
            # make prediction
            #testX = model_fit.predict(len(data), len(data))
            #print('Predicted next value of: ')
            #print(testX)
            print(model_fit.summary())
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def ARCH(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Double(dataset)
            ds = Preprocessing.Scaler_Standard(ds, 'ARCH', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data_ARCH(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            trainX = pd.DataFrame(trainX)
            n_test = 10
            print('Starting to fit model: ARCH')
            # Fit model
            model = arch_model(trainX, mean='Zero', vol='ARCH', p=15)
            model_fit = model.fit()
            # Make prediction
            #testX = model_fit.forecast(horizon=n_test)
            #print('Predicted next value of: ')
            print(model_fit.summary())
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def GARCH(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Double(dataset)
            ds = Preprocessing.Scaler_Standard(ds, 'GARCH', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data_ARCH(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            # trainX = pd.DataFrame(trainX)
            print('Starting to fit model: GARCH')
            # Fit model
            model = arch_model(trainX, mean='Zero', vol='GARCH', p=15, q=15)
            model_fit = model.fit()
            print(model_fit.summary())
            # make predictions
            print(trainX.shape)
            trainPredict = model_fit.forecast(horizon=len(trainX))
            print(trainPredict.mean)
            print(trainY.shape)
            print(testX.shape)
            testPredict = model_fit.forecast(horizon=len(testX))
            # invert predictions
            trainPredict = Preprocessing.Invert_Transform(trainPredict.mean, coin, 'GARCH', 'SCALERSTANDARD')
            print(trainPredict.shape)
            print(trainPredict)
            trainY = Preprocessing.Invert_Transform([trainY], coin, 'GARCH', 'SCALERSTANDARD')
            print(trainY[0].shape)
            testPredict = Preprocessing.Invert_Transform(testPredict.mean, coin, 'GARCH', 'SCALERSTANDARD')
            print(testPredict.shape)
            testY = Preprocessing.Invert_Transform([testY], coin, 'GARCH', 'SCALERSTANDARD')
            print(testY[0].shape)
            ds = Preprocessing.Invert_Transform(ds, coin, 'GARCH', 'SCALERSTANDARD')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'GARCH')
            # Saving model to disk
            ModelParameters.Save_Model(model_fit, coin, 'GARCH')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True
