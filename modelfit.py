# manually search hyperparameters for classification
from numpy import mean
from numpy.random import randn
from numpy.random import rand
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from dataprocessing import DataProcessing
import matplotlib.pyplot as plt
from finance import Finance
from machinelearning import MachineLearning


class ModelFit:
		# objective function
		def Objective(model, trainX, trainY): # , cfg):
			try:
				# unnpack config
				# eta, alpha = cfg
				# define model
				# model = Perceptron(penalty='elasticnet', alpha=alpha, eta0=eta)
				# define evaluation procedure
				cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
				# evaluate model
				scores = cross_val_score(model, trainX, trainY, scoring='accuracy', cv=cv, n_jobs=-1)
				# calculate mean accuracy
				result = mean(scores)
			except Exception as e:
				print("An exception occurred - {}".format(e))
			return result


		# take a step in the search space
		def Step(cfg, step_size):
			try:
				# unpack the configuration
				eta, alpha = cfg
				# step eta
				new_eta = eta + randn() * step_size
				# check the bounds of eta
				if new_eta <= 0.0:
					new_eta = 1e-8
				# step alpha
				new_alpha = alpha + randn() * step_size
				# check the bounds of alpha
				if new_alpha < 0.0:
					new_alpha = 0.0
				# return the new configuration
			except Exception as e:
				print("An exception occurred - {}".format(e))
			return [new_eta, new_alpha]


		# hill climbing local search algorithm
		def Hillclimbing(trainX, trainY, objective, n_iter, step_size):
			try:
				# starting point for the search
				solution = [rand(), rand()]
				# evaluate the initial point
				solution_eval = objective(trainX, trainY, solution)
				# run the hill climb
				for i in range(n_iter):
					# take a step
					candidate = ModelFit.Step(solution, step_size)
					# evaluate candidate point
					candidate_eval = objective(trainX, trainY, candidate)
					# check if we should keep the new point
					if candidate_eval >= solution_eval:
						# store the new point
						solution, solution_eval = candidate, candidate_eval
						# report progress
						print('>%d, cfg=%s %.5f' % (i, solution, solution_eval))
			except Exception as e:
				print("An exception occurred - {}".format(e))
			return [solution, solution_eval]


		# define dataset
		def Calculate(dataset, coin):
			try:
				trainX, trainY, testX, testY, ds = DataProcessing.Preprocess_Data(dataset, 'LR', coin) # ???
				# define the total iterations
				n_iter = 100
				# step size in the search space
				step_size = 0.1
				# perform the hill climbing search
				cfg, score = ModelFit.Hillclimbing(trainX, trainY, ModelFit.Objective, n_iter, step_size)
				print('Done!')
				print('cfg=%s: Mean Accuracy: %f' % (cfg, score))
			except Exception as e:
				print("An exception occurred - {}".format(e))
			return True
		# unir iterate con config

		def Iterate(dataset, coin):
			try:
				# prepare models
				models = []
				print('Estimating results for Multi-Layer Perceptron')
				models.append(('MLP', MachineLearning.MLP(dataset, coin, cfg)))
				print('Estimatings results for Convolutional Neural Networks')
				models.append(('CNN', MachineLearning.CNN(dataset, coin)))
				print('Estimatings results for Recurrent Neural Networks')
				models.append(('RNN', MachineLearning.RNN(dataset, coin)))
				print('Estimating results for Long Short Term Memory')
				models.append(('LSTM', 'MachineLearning.LSTM'))
				print('Estimating results for Auto Regressive')
				models.append(('AR', Finance.AR(dataset, 1)))
				# Evaluate each model in turn
				results = []
				names = []
				scoring = 'accuracy'
				for name, model in models:
					print('Starting to calculate the score for the model: ' + model)
					cv_results = ModelFit.Calculate(dataset, coin, model, cfg) # por ultimo hacerla con IFS
					results.append(cv_results)
					names.append(name)
					msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
					print(msg)
				# boxplot algorithm comparison
				fig = plt.figure()
				fig.suptitle('Algorithm Comparison')
				ax = fig.add_subplot(111)
				plt.boxplot(results)
				ax.set_xticklabels(names)
				plt.show()
			except Exception as e:
				print("An exception occurred - {}".format(e))
			return names, results
