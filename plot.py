import matplotlib.pyplot as plt

class ModelPlot:
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()






# shift train predictions for plotting
print('Shifting positions for the training set to plot correctly.')
trainPredictPlot = numpy.empty_like(ds)
to_append = numpy.array([[0]])
for x in range(TO_FORECAST):
    trainPredictPlot = numpy.append(trainPredictPlot, to_append, 0)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[LOOK_BACK:len(trainPredict) + LOOK_BACK, :] = trainPredict

# shift test predictions for plotting
print('Shifting positions for the test set to plot correctly.')
testPredictPlot = numpy.empty_like(ds)
to_append = numpy.array([[0]])
for x in range(TO_FORECAST):
    testPredictPlot = numpy.append(testPredictPlot, to_append, 0)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (LOOK_BACK * 2) + 1:len(ds) - 1 + TO_FORECAST, :] = testPredict

# plot baseline and predictions
plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(ds), color='blue', label='Set')
plt.plot(trainPredictPlot, color='green')
plt.plot(testPredictPlot, color='orange')
plt.title('Close Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(color='k', linestyle='dotted', linewidth=1)
plt.show()
except Exception as e:
print("An exception occurred - {}".format(e))
return False
return True

# plot baseline and predictions
# Load scaler into new model
scaler = pickle.load(open('scaler_lstm.pkl', 'rb'))
print('Scaler loaded correctly')
plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(ds), color='blue', label='Set')
plt.plot(testX, color='green')
plt.plot(Yhat, color='orange')
plt.title('Close Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(color='k', linestyle='dotted', linewidth=1)
plt.show()