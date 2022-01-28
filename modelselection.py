import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy

from finance import Finance
from machinelearning import MachineLearning


# prepare configuration for cross validation test harness
seed = 7
numpy.random.seed(seed)

class ModelSelection:
	def Calculate(dataset):
		try:
			# prepare models
			models = []
			print('estimating results for LR')
			models.append(('LR', MachineLearning.Logistic_Regression(dataset)))
			print('estimating results for svm')
			models.append(('SVM', MachineLearning.SVC(dataset)))
			models.append(('KNN', MachineLearning.K_Neighbors_Classifier(dataset)))
			models.append(('CART', MachineLearning.Decision_Tree(dataset)))
			models.append(('NB', MachineLearning.GaussianNB(dataset)))
			models.append(('PN', MachineLearning.Perceptron(dataset)))
			models.append(('SVC', MachineLearning.Linear_SVC(dataset)))
			models.append(('SGD', MachineLearning.SGD_Classifier(dataset)))
			models.append(('DT', MachineLearning.Decision_Tree(dataset)))
			models.append(('RF', MachineLearning.Random_Forest(dataset)))
			models.append(('LSTM', MachineLearning.LSTM(dataset)))
			models.append(('RNN', MachineLearning.RNN(dataset)))
			models.append(('AR', Finance.AR(dataset, 1)))
			models.append(('ARIMA', Finance.ARIMA(dataset)))
			models.append(('SARIMA', Finance.SARIMA(dataset)))
			models.append(('SARIMAX', Finance.SARIMAX(dataset)))
			models.append(('VAR', Finance.VAR(dataset)))
			models.append(('VARMA', Finance.VARMA(dataset)))
			models.append(('VARMAX', Finance.VARMAX(dataset)))
			models.append(('SES', Finance.SES(dataset)))
			models.append(('HWES', Finance.HWES(dataset)))
			models.append(('ARCH', Finance.ARCH(dataset)))
			models.append(('GARCH', Finance.GARCH(dataset)))

			# Evaluate each model in turn
			results = []
			names = []
			scoring = 'accuracy'
			for name, model in models:
				print('Starting to calculate the score for the model: ' + model)
				kfold = model_selection.KFold(n_splits=2, random_state=seed)
				cv_results = model_selection.cross_val_score(model, dataset, cv=kfold, scoring=scoring)
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

			#models = pd.DataFrame({
			#	'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
			#			  'Random Forest', 'Naive Bayes', 'Perceptron',
			#			  'Stochastic Gradient Decent', 'Linear SVC',
			#			  'Decision Tree'],
			#	'Score': [acc_svc, acc_knn, acc_log,
			#			  acc_random_forest, acc_gaussian, acc_perceptron,
			#			  acc_sgd, acc_linear_svc, acc_decision_tree]})
			#models.sort_values(by='Score', ascending=False)




		except Exception as e:
			print("An exception occurred - {}".format(e))
			return False
		return True
