import matplotlib.pyplot as plt

class ModelPlot:


pred0 = model_fit.get_prediction(start=0, dynamic=False)
pred0_ci = pred0.conf_int()
pred1 = model_fit.get_prediction(start=0, dynamic=True)
pred1_ci = pred1.conf_int()
pred2 = model_fit.get_forecast(steps=predict)
pred2_ci = pred2.conf_int()
pred0 = pd.DataFrame(pred0.predicted_mean)
pred1 = pd.DataFrame(pred1.predicted_mean)
pred2 = pd.DataFrame(pred2.predicted_mean)

ax = df1.plot(figsize=(20, 16))
pred0.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
pred1.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
pred2.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
# ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('Close Price')
plt.xlabel('Date')
plt.legend()
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