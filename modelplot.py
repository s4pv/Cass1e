import matplotlib.pyplot as plt
import numpy
import pandas as pd
import warnings
import os
from helper import Helper

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

N_STEPS_IN = parsed_config['ml_options']['N_STEPS_IN']
N_STEPS_OUT = parsed_config['ml_options']['N_STEPS_OUT']

class ModelPlot:
    def Shift_Plot(ds, trainPredict, testPredict):
        try:
            # shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(ds)
            trainPredictPlot[:, :] = numpy.nan
            trainPredictPlot[N_STEPS_IN:len(trainPredict) + N_STEPS_IN, :] = trainPredict
            # shift test predictions for plotting
            testPredictPlot = numpy.empty_like(ds)
            testPredictPlot[:, :] = numpy.nan
            testPredictPlot[len(trainPredict) + (N_STEPS_IN * 2) + 1:len(ds) - 1, :] = testPredict
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return trainPredictPlot, testPredictPlot

    def Shift_Forecast_Plot(forecastDB, trainPredict, forecast):
        try:
            # shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(forecastDB)
            to_append = numpy.array([[0]])
            for x in range(N_STEPS_OUT):
                trainPredictPlot = numpy.append(trainPredictPlot, to_append, 0)
            trainPredictPlot[:, :] = numpy.nan
            trainPredictPlot[N_STEPS_IN:len(trainPredict) + N_STEPS_IN, :] = trainPredict
            # shift test predictions for plotting
            forecastPlot = numpy.empty_like(forecast)
            to_append = numpy.array([[0]])
            for x in range(N_STEPS_OUT):
                forecastPlot = numpy.append(forecastPlot, to_append, 0)
            forecastPlot[:, :] = numpy.nan
            forecastPlot[len(trainPredict) + (N_STEPS_IN * 2) + 1:len(forecastDB) - 1 + N_STEPS_OUT, :] = forecast
            # append # of predictions to the whole database for plotting
            forecastDBPlot = numpy.empty_like(forecastDB)
            to_append = numpy.array([[0]])
            for x in range(N_STEPS_OUT):
                forecastDBPlot = numpy.append(forecastDB, to_append, 0)
            forecastDBPlot[:, :] = numpy.nan
            forecastDBPlot[N_STEPS_IN:len(forecastDB) - 1 + N_STEPS_OUT, :] = forecastDB
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return forecastDB, trainPredictPlot, forecast

    def Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, modelname):
        try:
            # plot baseline and predictions
            plt.figure(figsize=(10, 6))
            plt.plot(ds, color='blue', label='Set')
            plt.plot(trainPredictPlot, color='green')
            plt.plot(testPredictPlot, color='orange')
            plt.title('Close Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(color='k', linestyle='dotted', linewidth=1)
            #plt.show()
            filename = os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/model_plots',
                                    str(coin['symbol']) + '_model_' + str(modelname) + '.png')
            plt.savefig(filename)
            plt.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Plot_Forecast(ds, forecastPlot, coin, modelname):
        try:
            # plot baseline and predictions
            plt.figure(figsize=(10, 6))
            plt.plot(ds, color='blue', label='Set')
            plt.plot(forecastPlot, color='red')
            plt.title('Close Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(color='k', linestyle='dotted', linewidth=1)
            #plt.show()
            filename = os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/model_plots',
                                    str(coin['symbol']) + '_model_' + str(modelname) + '.png')
            plt.savefig(filename)
            plt.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True