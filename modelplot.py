import matplotlib.pyplot as plt
import numpy
import warnings
import os
from helper import Helper

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

N_STEPS_IN = parsed_config['ml_options']['N_STEPS_IN']
N_STEPS_OUT = parsed_config['ml_options']['N_STEPS_OUT']

class ModelPlot:
    def Shift_Plot(ds, trainLen, trainPredict, testPredict):
        try:
            # shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(ds)
            trainPredictPlot[:, :] = numpy.nan
            #print(trainPredict.shape)
            trainPredict = numpy.reshape(trainPredict, (N_STEPS_OUT, 1))
            #print(trainPredict.shape)
            trainPredictPlot[N_STEPS_IN + trainLen -1:len(trainPredict) + N_STEPS_IN + trainLen -1, :] = trainPredict
            # shift test predictions for plotting
            testPredictPlot = numpy.empty_like(ds)
            testPredictPlot[:, :] = numpy.nan
            #print(testPredict.shape)
            testPredict = numpy.reshape(testPredict, (N_STEPS_OUT, 1))
            #print(testPredict.shape)
            testPredictPlot[len(ds) - 1 - N_STEPS_OUT:len(ds) - 1, :] = testPredict
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return trainPredictPlot, testPredictPlot

    def Shift_Forecast_Plot(close, forecast):
        try:
            # shift forecast predictions for plotting
            forecastPlot = numpy.empty_like(close)
            to_append = numpy.array([[0]])
            #print(forecastPlot.shape)
            for x in range(N_STEPS_OUT):
                forecastPlot = numpy.append(forecastPlot, to_append, 0)
            forecastPlot[:, :] = numpy.nan
            #print(forecastPlot.shape)
            #print(N_STEPS_IN + len(forecastPlot) - 1 - N_STEPS_OUT)
            #print(N_STEPS_IN + len(forecastPlot) - 1)
            #print(forecast.shape)
            forecast = numpy.reshape(forecast, (N_STEPS_OUT, 1))
            #print(forecast.shape)
            forecastPlot[N_STEPS_IN + len(forecastPlot) - 1 - N_STEPS_OUT:N_STEPS_IN + len(forecastPlot) - 1, :] = forecast
            # expand close predictions for plotting
            closePlot = numpy.empty_like(close)
            to_append = numpy.array([[0]])
            #print(closePlot.shape)
            for x in range(N_STEPS_OUT):
                closePlot = numpy.append(closePlot, to_append, 0)
            closePlot[:, :] = numpy.nan
            #print(closePlot.shape)
            #print(len(closePlot) + N_STEPS_IN - N_STEPS_OUT - 1)
            #print(close.shape)
            closePlot[N_STEPS_IN:len(closePlot) + N_STEPS_IN - N_STEPS_OUT, :] = close
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return forecastPlot, closePlot

    def Plot_Actual(ds, trainYPlot, testYPlot, trainPredictPlot, testPredictPlot, coin, modelname):
        try:
            # plot baseline and predictions
            plt.figure(figsize=(10, 6))
            plt.plot(ds, color='gray', label='Set')
            plt.plot(trainPredictPlot, color='orange', label='trainPredict', linestyle='dashed')
            plt.plot(testPredictPlot, color='blue', label='testPredict', linestyle='dashed')
            plt.plot(trainYPlot, color='yellow', label='trainY')
            plt.plot(testYPlot, color='dodgerblue', label='testY')
            plt.title('Close Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend(loc='lower right')
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

    def Plot_Forecast(cPlot, forecastPlot, coin, modelname):
        try:
            # plot baseline and predictions
            plt.figure(figsize=(10, 6))
            plt.plot(cPlot, color='gray', label='Set')
            plt.plot(forecastPlot, color='red', label='Forecast', linestyle='dashed')
            plt.title('Close Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend(loc='lower right')
            plt.grid(color='k', linestyle='dotted', linewidth=1)
            #plt.show()
            filename = os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/model_forecasts_plots',
                                    str(coin['symbol']) + '_forecast_model_' + str(modelname) + '.png')
            plt.savefig(filename)
            plt.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True