from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import math
import warnings
from sklearn.metrics import mean_squared_error
from helper import Helper

import numpy
import matplotlib.pyplot as plt

from preprocessing import Preprocessing

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

TO_FORECAST = parsed_config['forecast_options']['TO_FORECAST']

NO_DAYS = parsed_config['model_options']['NO_DAYS']

LOOK_BACK = parsed_config['ml_options']['LOOK_BACK']
TRAIN_SIZE = parsed_config['ml_options']['TRAIN_SIZE']
NEURONS = parsed_config['ml_options']['NEURONS']
DROPOUT = parsed_config['ml_options']['DROPOUT']
BATCH_SIZE = parsed_config['ml_options']['BATCH_SIZE']
OUTPUT_DIM = parsed_config['ml_options']['OUTPUT_DIM']
EPOCH = parsed_config['ml_options']['EPOCH']
N_FEATURES = parsed_config['ml_options']['N_FEATURES']
HIDDEN_UNITS = parsed_config['ml_options']['HIDDEN_UNITS']
TIME_STEPS = parsed_config['ml_options']['TIME_STEPS']
DENSE_UNITS = parsed_config['ml_options']['DENSE_UNITS']
FILTERS = parsed_config['ml_options']['FILTERS']
KERNEL_SIZE = parsed_config['ml_options']['KERNEL_SIZE']
POOL_SIZE = parsed_config['ml_options']['POOL_SIZE']
VERBOSE = parsed_config['ml_options']['VERBOSE']


class MachineLearning:
    def Logistic_Regression(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'LR', coin)
            logreg = LogisticRegression()
            print('starting the fit')
            print(trainX.shape)
            print(trainY.shape)
            print(trainX)
            print(trainY)
            logreg.fit(trainX, trainY)
            #yhat = logreg.predict(testX)
            acc_log = round(logreg.score(trainX, trainY) * 100, 2)
            #print(yhat)
            print(acc_log)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def SVC(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'SVC', coin)
            svc = SVC()
            svc.fit(trainX, trainY)
            #Yhat = svc.predict(testX)
            acc_svc = round(svc.score(trainX, trainY) * 100, 2)
            print(acc_svc)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def K_Neighbors_Classifier(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'KNN', coin)
            knn = KNeighborsClassifier()
            knn.fit(trainX, trainY)
            #Yhat = knn.predict(testX)
            acc_knn = round(knn.score(trainX, trainY) * 100, 2)
            print(acc_knn)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def GaussianNB(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'NB', coin)
            gaussian = GaussianNB()
            print('gauss')
            gaussian.fit(trainX, trainY)
            #Yhat = gaussian.predict(testX)
            acc_gauss = round(gaussian.score(trainX, trainY) * 100, 2)
            print(acc_gauss)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Perceptron(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'PN', coin)
            perceptron = Perceptron()
            perceptron.fit(trainX, trainY)
            #Yhat = perceptron.predict(testX)
            acc_perc = round(perceptron.score(trainX, trainY) * 100, 2)
            print(acc_perc)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Linear_SVC(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'SVC', coin)
            linear_SVC = LinearSVC()
            linear_SVC.fit(trainX, trainY)
            #Yhat = linear_SVC.predict(testX)
            acc_l_svc = round(linear_SVC.score(trainX, trainY) * 100, 2)
            print(acc_l_svc)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def SGD_Classifier(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'SGD', coin)
            sgd = SGDClassifier()
            sgd.fit(trainX, trainY)
            #Yhat = sgd.predict(testX)
            acc_sgd = round(sgd.score(trainX, trainY) * 100, 2)
            print(acc_sgd)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Decision_Tree(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'CART', coin)
            decision_tree = DecisionTreeClassifier()
            decision_tree.fit(trainX, trainY)
            #Yhat = decision_tree.predict(testX)
            acc_dt = round(decision_tree.score(trainX, trainY) * 100, 2)
            print(acc_dt)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Random_Forest(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'RF', coin)
            random_forest = RandomForestClassifier()
            random_forest.fit(trainX, trainY)
            #Yhat = random_forest.predict(testX)
            acc_rf = round(random_forest.score(trainX, trainY) * 100, 2)
            print(acc_rf)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def MLP(dataset, coin, n_feat, epoch, batch_size, verbose):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'MLP', coin)
            # Create and fit the Multi Layer Perceptron.
            mlp = Sequential()
            # Memory between batches -> batch_input_shape=(samples, time steps, n_features)
            mlp.add(Dense(NEURONS, activation='relu', input_dim=N_FEATURES))
            mlp.add(Dense(1))
            mlp.compile(loss='mean_squared_error', optimizer='adam')
            # Memory between batches -> epoch can be added in a for
            mlp.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE, shuffle=False)
            #mlp.reset_states()
            #print('mlp')
            #Yhat = mlp.predict(testX)
            #acc_mlp = round(mlp.score(trainX, trainY) * 100, 2)
            #print(acc_mlp)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def CNN(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'CNN', coin)
            # Create and fit the Convolutional Neural  network.
            cnn = Sequential()
            # Memory between batches -> batch_input_shape=(samples, time steps, n_features)
            cnn.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu', input_shape=(TIME_STEPS,
                                                                                                     N_FEATURES)))
            cnn.add(MaxPooling1D(pool_size=POOL_SIZE))
            cnn.add(Flatten())
            cnn.add(Dense(DENSE_UNITS*10, activation='relu'))
            cnn.add(Dense(OUTPUT_DIM))
            cnn.compile(loss='mean_squared_error', optimizer='adam')
            # Memory between batches -> epoch can be added in a for
            cnn.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE, shuffle=False)
            #cnn.reset_states()
            #print('cnn')
            #Yhat = cnn.predict(testX)
            #acc_cnn = round(cnn.score(trainX, trainY) * 100, 2)
            #print(acc_cnn)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def RNN(dataset, coin):
        try:
            trainX, trainY, testX, testY, ds = Preprocessing.Preprocess_Data(dataset, 'RNN', coin)
            # Create and fit the Recurrent Neural network.
            rnn = Sequential()
            # Memory between batches -> batch_input_shape=(samples, time steps, n_features)
            rnn.add(SimpleRNN(HIDDEN_UNITS, input_shape=(TIME_STEPS, 1), activation='tanh'))
            rnn.add(Dense(units=DENSE_UNITS, activation='tanh'))
            rnn.compile(loss='mean_squared_error', optimizer='adam')
            # Train the network
            rnn.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE)
            #print('rnn')
            #Yhat = rnn.predict(testX)
            #acc_rnn = round(rnn.score(trainX, trainY) * 100, 2)
            #print(acc_rnn)
            # Evalute model
            #train_mse = rnn.evaluate(trainX, trainY)
            #test_mse = rnn.evaluate(testX, testY)
            # Print error
            #print("Train set MSE = ", train_mse)
            #print("Test set MSE = ", test_mse)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def LSTM(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Float(dataset)
            ds = Preprocessing.Minmax_Scaler(ds, 'LSTM', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            # Create and fit the Long short Term Memory network.
            lstm = Sequential()
            # Memory between batches -> batch_input_shape=(samples, time steps, n_features)
            lstm.add(LSTM(NEURONS, batch_input_shape=(BATCH_SIZE, LOOK_BACK, N_FEATURES), stateful=True, return_sequences=True))
            lstm.add(LSTM(NEURONS, batch_input_shape=(BATCH_SIZE, LOOK_BACK, N_FEATURES), stateful=True))
            lstm.add(Dropout(DROPOUT))
            lstm.add(Dense(OUTPUT_DIM))
            lstm.compile(loss='mean_squared_error', optimizer='adam')
            # Memory between batches -> epoch can be added in a for
            lstm.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE, shuffle=False)
            print(lstm.summary())
            lstm.reset_states()
            # make predictions
            print(trainX.shape)
            trainPredict = lstm.predict(trainX)
            lstm.reset_states()
            print(trainY.shape)
            testPredict = lstm.predict(testX)
            # invert predictions
            trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'LSTM', 'MINMAXSCALER')
            print(trainPredict.shape)
            trainY = Preprocessing.Invert_Transform([trainY], coin, 'LSTM', 'MINMAXSCALER')
            print(trainY[0].shape)
            testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'LSTM', 'MINMAXSCALER')
            print(testPredict.shape)
            testY = Preprocessing.Invert_Transform([testY], coin, 'LSTM', 'MINMAXSCALER')
            print(testY[0].shape)
            # Estimate model performance
            trainScore = lstm.evaluate(trainPredict[:,0], trainY[0], verbose=0)
            print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
            testScore = lstm.evaluate(testPredict[:,0], testY[0], verbose=0)
            print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
            # shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(ds)
            trainPredictPlot[:, :] = numpy.nan
            trainPredictPlot[LOOK_BACK:len(trainPredict) + LOOK_BACK, :] = trainPredict
            # shift test predictions for plotting
            testPredictPlot = numpy.empty_like(ds)
            testPredictPlot[:, :] = numpy.nan
            testPredictPlot[len(trainPredict) + (LOOK_BACK * 2) + 1:len(dataset) - 1, :] = testPredict
            # plot baseline and predictions
            plt.plot(Preprocessing.Invert_Transform(ds, coin, 'LSTM', 'MINMAXSCALER'), label='Actual')
            plt.plot(trainPredictPlot, label='Training')
            plt.plot(testPredictPlot, label='Testing')
            plt.show()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True
