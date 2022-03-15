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

from preprocessing import Preprocessing
from modelplot import ModelPlot
from modelparameters import ModelParameters

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
            ds = Preprocessing.Reshape_Int(dataset)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            logreg = LogisticRegression()
            print('starting the fit')
            logreg.fit(trainX, trainY)
            # make predictions
            print(trainX.shape)
            trainPredict = logreg.predict(trainX)
            #logreg.reset_states()
            print(trainY.shape)
            testPredict = logreg.predict(testX)
            # invert predictions
            #trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'Logistic_Regression', 'SCALERSTANDARD')
            print(trainPredict.shape)
            #trainY = Preprocessing.Invert_Transform([trainY], coin, 'Logistic_Regression', 'SCALERSTANDARD')
            print(trainY[0].shape)
            #testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'Logistic_Regression', 'SCALERSTANDARD')
            print(testPredict.shape)
            #testY = Preprocessing.Invert_Transform([testY], coin, 'Logistic_Regression', 'SCALERSTANDARD')
            print(testY[0].shape)
            #ds = Preprocessing.Invert_Transform(ds, coin, 'Logistic_Regression', 'SCALERSTANDARD')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'Logistic_Regression')
            # Saving model to disk
            ModelParameters.Save_Model(logreg, coin, 'Logistic_Regression')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def SVC(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Int(dataset)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            svc = SVC()
            svc.fit(trainX, trainY)
            # make predictions
            print(trainX.shape)
            trainPredict = svc.predict(trainX)
            #svc.reset_states()
            print(trainY.shape)
            testPredict = svc.predict(testX)
            # invert predictions
            #trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'SVC', 'SCALERSTANDARD')
            print(trainPredict.shape)
            #trainY = Preprocessing.Invert_Transform([trainY], coin, 'SVC', 'SCALERSTANDARD')
            print(trainY[0].shape)
            #testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'SVC', 'SCALERSTANDARD')
            print(testPredict.shape)
            #testY = Preprocessing.Invert_Transform([testY], coin, 'SVC', 'SCALERSTANDARD')
            print(testY[0].shape)
            #ds = Preprocessing.Invert_Transform(ds, coin, 'SVC', 'SCALERSTANDARD')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'SVC')
            # Saving model to disk
            ModelParameters.Save_Model(svc, coin, 'SVC')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def K_Neighbors_Classifier(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Int(dataset)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            knn = KNeighborsClassifier()
            knn.fit(trainX, trainY)
            # make predictions
            print(trainX.shape)
            trainPredict = knn.predict(trainX)
            #knn.reset_states()
            print(trainY.shape)
            testPredict = knn.predict(testX)
            # invert predictions
            #trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'K_Neighbors_Classifier', 'SCALERSTANDARD')
            print(trainPredict.shape)
            #trainY = Preprocessing.Invert_Transform([trainY], coin, 'K_Neighbors_Classifier', 'SCALERSTANDARD')
            print(trainY[0].shape)
            #testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'K_Neighbors_Classifier', 'SCALERSTANDARD')
            print(testPredict.shape)
            #testY = Preprocessing.Invert_Transform([testY], coin, 'K_Neighbors_Classifier', 'SCALERSTANDARD')
            print(testY[0].shape)
            #ds = Preprocessing.Invert_Transform(ds, coin, 'K_Neighbors_Classifier', 'SCALERSTANDARD')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'K_Neighbors_Classifier')
            # Saving model to disk
            ModelParameters.Save_Model(knn, coin, 'K_Neighbors_Classifier')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def GaussianNB(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Int(dataset)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            gaussian = GaussianNB()
            print('gauss')
            gaussian.fit(trainX, trainY)
            # make predictions
            print(trainX.shape)
            trainPredict = gaussian.predict(trainX)
            #gaussian.reset_states()
            print(trainY.shape)
            testPredict = gaussian.predict(testX)
            # invert predictions
            #trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'GaussianNB', 'SCALERSTANDARD')
            print(trainPredict.shape)
            #trainY = Preprocessing.Invert_Transform([trainY], coin, 'GaussianNB', 'SCALERSTANDARD')
            print(trainY[0].shape)
            #testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'GaussianNB', 'SCALERSTANDARD')
            print(testPredict.shape)
            #testY = Preprocessing.Invert_Transform([testY], coin, 'GaussianNB', 'SCALERSTANDARD')
            print(testY[0].shape)
            #ds = Preprocessing.Invert_Transform(ds, coin, 'GaussianNB', 'SCALERSTANDARD')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'GaussianNB')
            # Saving model to disk
            ModelParameters.Save_Model(gaussian, coin, 'GaussianNB')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Perceptron(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Int(dataset)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            perceptron = Perceptron()
            perceptron.fit(trainX, trainY)
            # make predictions
            print(trainX.shape)
            trainPredict = perceptron.predict(trainX)
            #perceptron.reset_states()
            print(trainY.shape)
            testPredict = perceptron.predict(testX)
            # invert predictions
            #trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'Perceptron', 'SCALERSTANDARD')
            print(trainPredict.shape)
            #trainY = Preprocessing.Invert_Transform([trainY], coin, 'Perceptron', 'SCALERSTANDARD')
            print(trainY[0].shape)
            #testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'Perceptron', 'SCALERSTANDARD')
            print(testPredict.shape)
            #testY = Preprocessing.Invert_Transform([testY], coin, 'Perceptron', 'SCALERSTANDARD')
            print(testY[0].shape)
            #ds = Preprocessing.Invert_Transform(ds, coin, 'Perceptron', 'SCALERSTANDARD')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'Perceptron')
            # Saving model to disk
            ModelParameters.Save_Model(perceptron, coin, 'Perceptron')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Linear_SVC(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Int(dataset)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            linear_SVC = LinearSVC()
            linear_SVC.fit(trainX, trainY)
            # make predictions
            print(trainX.shape)
            trainPredict = linear_SVC.predict(trainX)
            #linear_SVC.reset_states()
            print(trainY.shape)
            testPredict = linear_SVC.predict(testX)
            # invert predictions
            #trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'Linear_SVC', 'SCALERSTANDARD')
            print(trainPredict.shape)
            #trainY = Preprocessing.Invert_Transform([trainY], coin, 'Linear_SVC', 'SCALERSTANDARD')
            print(trainY[0].shape)
            #testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'Linear_SVC', 'SCALERSTANDARD')
            print(testPredict.shape)
            #testY = Preprocessing.Invert_Transform([testY], coin, 'Linear_SVC', 'SCALERSTANDARD')
            print(testY[0].shape)
            #ds = Preprocessing.Invert_Transform(ds, coin, 'Linear_SVC', 'SCALERSTANDARD')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'Linear_SVC')
            # Saving model to disk
            ModelParameters.Save_Model(linear_SVC, coin, 'Linear_SVC')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def SGD_Classifier(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Int(dataset)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            sgd = SGDClassifier()
            sgd.fit(trainX, trainY)
            # make predictions
            print(trainX.shape)
            trainPredict = sgd.predict(trainX)
            #sgd.reset_states()
            print(trainY.shape)
            testPredict = sgd.predict(testX)
            # invert predictions
            #trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'SGD_Classifier', 'SCALERSTANDARD')
            print(trainPredict.shape)
            #trainY = Preprocessing.Invert_Transform([trainY], coin, 'SGD_Classifier', 'SCALERSTANDARD')
            print(trainY[0].shape)
            #testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'SGD_Classifier', 'SCALERSTANDARD')
            print(testPredict.shape)
            #testY = Preprocessing.Invert_Transform([testY], coin, 'SGD_Classifier', 'SCALERSTANDARD')
            print(testY[0].shape)
            #ds = Preprocessing.Invert_Transform(ds, coin, 'SGD_Classifier', 'SCALERSTANDARD')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'SGD_Classifier')
            # Saving model to disk
            ModelParameters.Save_Model(sgd, coin, 'SGD_Classifier')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Decision_Tree(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Int(dataset)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            decision_tree = DecisionTreeClassifier()
            decision_tree.fit(trainX, trainY)
            # make predictions
            print(trainX.shape)
            trainPredict = decision_tree.predict(trainX)
            #decision_tree.reset_states()
            print(trainY.shape)
            testPredict = decision_tree.predict(testX)
            # invert predictions
            #trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'Decision_Tree', 'SCALERSTANDARD')
            print(trainPredict.shape)
            #trainY = Preprocessing.Invert_Transform([trainY], coin, 'Decision_Tree', 'SCALERSTANDARD')
            print(trainY[0].shape)
            #testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'Decision_Tree', 'SCALERSTANDARD')
            print(testPredict.shape)
            #testY = Preprocessing.Invert_Transform([testY], coin, 'Decision_Tree', 'SCALERSTANDARD')
            print(testY[0].shape)
            #ds = Preprocessing.Invert_Transform(ds, coin, 'Decision_Tree', 'SCALERSTANDARD')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'Decision_Tree')
            # Saving model to disk
            ModelParameters.Save_Model(decision_tree, coin, 'Decision_Tree')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Random_Forest(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Int(dataset)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            random_forest = RandomForestClassifier()
            random_forest.fit(trainX, trainY)
            # make predictions
            print(trainX.shape)
            trainPredict = random_forest.predict(trainX)
            #random_forest.reset_states()
            print(trainY.shape)
            testPredict = random_forest.predict(testX)
            # invert predictions
            #trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'Random_Forest', 'SCALERSTANDARD')
            print(trainPredict.shape)
            #trainY = Preprocessing.Invert_Transform([trainY], coin, 'Random_Forest', 'SCALERSTANDARD')
            print(trainY[0].shape)
            #testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'Random_Forest', 'SCALERSTANDARD')
            print(testPredict.shape)
            #testY = Preprocessing.Invert_Transform([testY], coin, 'Random_Forest', 'SCALERSTANDARD')
            print(testY[0].shape)
            #ds = Preprocessing.Invert_Transform(ds, coin, 'Random_Forest', 'SCALERSTANDARD')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'Random_Forest')
            # Saving model to disk
            ModelParameters.Save_Model(random_forest, coin, 'Random_Forest')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def MLP(dataset, coin, n_feat, epoch, batch_size, verbose):
        try:
            ds = Preprocessing.Reshape_Float(dataset)
            ds = Preprocessing.Minmax_Scaler(ds, 'MLP', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            # Create and fit the Multi Layer Perceptron.
            mlp = Sequential()
            # Memory between batches -> batch_input_shape=(samples, time steps, n_features)
            mlp.add(Dense(NEURONS, activation='relu', input_dim=N_FEATURES))
            mlp.add(Dense(1))
            mlp.compile(loss='mean_squared_error', optimizer='adam')
            # Memory between batches -> epoch can be added in a for
            mlp.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE, shuffle=False)
            #mlp.reset_states()
            # Train the network
            mlp.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE)
            # make predictions
            print(trainX.shape)
            trainPredict = mlp.predict(trainX, batch_size=BATCH_SIZE)
            mlp.reset_states()
            print(trainY.shape)
            testPredict = mlp.predict(testX, batch_size=BATCH_SIZE)
            # invert predictions
            trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'MLP', 'MINMAXSCALER')
            print(trainPredict.shape)
            trainY = Preprocessing.Invert_Transform([trainY], coin, 'MLP', 'MINMAXSCALER')
            print(trainY[0].shape)
            testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'MLP', 'MINMAXSCALER')
            print(testPredict.shape)
            testY = Preprocessing.Invert_Transform([testY], coin, 'MLP', 'MINMAXSCALER')
            print(testY[0].shape)
            ds = Preprocessing.Invert_Transform(ds, coin, 'MLP', 'MINMAXSCALER')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'MLP')
            # Saving model to disk
            ModelParameters.Save_Model(mlp, coin, 'MLP')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def CNN(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Float(dataset)
            ds = Preprocessing.Minmax_Scaler(ds, 'CNN', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
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
            # Train the network
            cnn.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE)
            # make predictions
            print(trainX.shape)
            trainPredict = cnn.predict(trainX, batch_size=BATCH_SIZE)
            cnn.reset_states()
            print(trainY.shape)
            testPredict = cnn.predict(testX, batch_size=BATCH_SIZE)
            # invert predictions
            trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'CNN', 'MINMAXSCALER')
            print(trainPredict.shape)
            trainY = Preprocessing.Invert_Transform([trainY], coin, 'CNN', 'MINMAXSCALER')
            print(trainY[0].shape)
            testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'CNN', 'MINMAXSCALER')
            print(testPredict.shape)
            testY = Preprocessing.Invert_Transform([testY], coin, 'CNN', 'MINMAXSCALER')
            print(testY[0].shape)
            ds = Preprocessing.Invert_Transform(ds, coin, 'CNN', 'MINMAXSCALER')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'CNN')
            # Saving model to disk
            ModelParameters.Save_Model(cnn, coin, 'CNN')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def RNN(dataset, coin):
        try:
            ds = Preprocessing.Reshape_Float(dataset)
            ds = Preprocessing.Minmax_Scaler(ds, 'RNN', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            # Create and fit the Recurrent Neural network.
            rnn = Sequential()
            # Memory between batches -> batch_input_shape=(samples, time steps, n_features)
            rnn.add(SimpleRNN(HIDDEN_UNITS, input_shape=(TIME_STEPS, 1), activation='tanh'))
            rnn.add(Dense(units=DENSE_UNITS, activation='tanh'))
            rnn.compile(loss='mean_squared_error', optimizer='adam')
            # Train the network
            rnn.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE)
            # make predictions
            print(trainX.shape)
            trainPredict = rnn.predict(trainX, batch_size=BATCH_SIZE)
            rnn.reset_states()
            print(trainY.shape)
            testPredict = rnn.predict(testX, batch_size=BATCH_SIZE)
            # invert predictions
            trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'RNN', 'MINMAXSCALER')
            print(trainPredict.shape)
            trainY = Preprocessing.Invert_Transform([trainY], coin, 'RNN', 'MINMAXSCALER')
            print(trainY[0].shape)
            testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'RNN', 'MINMAXSCALER')
            print(testPredict.shape)
            testY = Preprocessing.Invert_Transform([testY], coin, 'RNN', 'MINMAXSCALER')
            print(testY[0].shape)
            ds = Preprocessing.Invert_Transform(ds, coin, 'RNN', 'MINMAXSCALER')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'RNN')
            # Saving model to disk
            ModelParameters.Save_Model(rnn, coin, 'RNN')
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
            for i in range(EPOCH):
                lstm.fit(trainX, trainY, epochs=1, batch_size=BATCH_SIZE, verbose=VERBOSE, shuffle=False)
                lstm.reset_states()
            print(lstm.summary())
            # make predictions
            print(trainX.shape)
            trainPredict = lstm.predict(trainX, batch_size=BATCH_SIZE)
            lstm.reset_states()
            print(trainY.shape)
            testPredict = lstm.predict(testX, batch_size=BATCH_SIZE)
            # invert predictions
            trainPredict = Preprocessing.Invert_Transform(trainPredict, coin, 'LSTM', 'MINMAXSCALER')
            print(trainPredict.shape)
            trainY = Preprocessing.Invert_Transform([trainY], coin, 'LSTM', 'MINMAXSCALER')
            print(trainY[0].shape)
            testPredict = Preprocessing.Invert_Transform(testPredict, coin, 'LSTM', 'MINMAXSCALER')
            print(testPredict.shape)
            testY = Preprocessing.Invert_Transform([testY], coin, 'LSTM', 'MINMAXSCALER')
            print(testY[0].shape)
            ds = Preprocessing.Invert_Transform(ds, coin, 'LSTM', 'MINMAXSCALER')
            # Estimate model performance
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            print('Test Score: %.2f RMSE' % (testScore))
            # shift train predictions for plotting
            trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'LSTM')
            # Saving model to disk
            ModelParameters.Save_Model(lstm, coin, 'LSTM')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True
