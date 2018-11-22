from utilities import *

@timeit
def test_ann(trainx, trainy, testx, testy):
	""" Train and test an artificial neural network
	"""
	input_size = len(trainx.iloc[0])
	output_size = len(np.unique(trainy))
	train = ClassificationDataSet(input_size, 1, nb_classes=output_size)
	for i in range(len(trainx.index)):
		train.addSample(trainx.iloc[i].values, trainy.iloc[i])

	test = ClassificationDataSet(input_size, 1, nb_classes=output_size)
	for i in range(len(testx.index)):
		test.addSample(testx.iloc[i].values, testy.iloc[i])
	train._convertToOneOfMany()
	test._convertToOneOfMany()

	print("Number of training patterns: ", len(train))
	print("Input and output dimensions: ", train.indim, train.outdim)
	print("First sample (input, target):")
	print(train["input"][0], train["target"][0])

	n_hidden = 3
	fnn = buildNetwork(train.indim, n_hidden, train.outdim)
	trainer = BackpropTrainer(
	    fnn, dataset=train, momentum=0.1, verbose=True, weightdecay=0.01)

	print("# hidden nodes: {}".format(n_hidden))
	for i in range(25):
	    trainer.trainEpochs(1)
	    trnresult = percentError(trainer.testOnClassData(), train["target"])
	    tstresult = percentError(
	        trainer.testOnClassData(dataset=test), test["target"])
	    print("epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult)
	pred = fnn.activateOnDataset(test)
	preds = [y.argmax() for y in pred]
	print(accuracy_score(preds, testy, normalize=True))

def net_data(trainx, trainy, testx, testy):
	input_size = len(trainx.iloc[0])
	output_size = len(np.unique(trainy))
	train = ClassificationDataSet(input_size, 1, nb_classes=output_size)
	for i in range(len(trainx.index)):
		train.addSample(trainx.iloc[i].values, trainy.iloc[i])

	test = ClassificationDataSet(input_size, 1, nb_classes=output_size)
	for i in range(len(testx.index)):
		test.addSample(testx.iloc[i].values, testy.iloc[i])

	train._convertToOneOfMany()
	test._convertToOneOfMany()

	print("Number of training patterns: ", len(train))
	print("Input and output dimensions: ", train.indim, train.outdim)
	print("First sample (input, target):")
	print(train["input"][0], train["target"][0])
	return train, test

def build_2net(input_size, output_size, n_hidden=[5, 3]):
	""" Build a 2 hidden layer network give the layer sizes. """
	# Create network and modules
	net = FeedForwardNetwork()
	inp = LinearLayer(input_size)
	h1 = SigmoidLayer(n_hidden[0])
	h2 = TanhLayer(n_hidden[1])
	outp = LinearLayer(output_size)
	# Add modules
	net.addOutputModule(outp)
	net.addInputModule(inp)
	net.addModule(h1)
	net.addModule(h2)
	# Create connections
	net.addConnection(FullConnection(inp, h1, inSliceTo=6))
	net.addConnection(FullConnection(inp, h2, inSliceFrom=6))
	net.addConnection(FullConnection(h1, h2))
	net.addConnection(FullConnection(h2, outp))
	# Finish up
	net.sortModules()
	return net

@timeit
def test_ann2(trainx, trainy, testx, testy, n_hidden=[5, 3], n_iter=25):
	""" Test and train a 2-hidden layer neural network, where the first layer is
	composed of sigmoid units and the second layer is composed of tanh units.
	n_hibben is a 2-element list of the sizes of the hidden layers, and n_iter
	is the number of epochs to stop at.
	"""
	train, test = net_data(trainx, trainy, testx, testy)
	input_size = len(trainx.iloc[0])
	output_size = len(np.unique(trainy))
	net = build_2net(input_size, output_size, n_hidden=n_hidden)
	# Train the network using back-propagation
	trainer = BackpropTrainer(net, dataset=train, momentum=0.0, verbose=True, weightdecay=0.0)

	for i in range(n_iter):
		trainer.trainEpochs(1)
		trnresult = percentError(trainer.testOnClassData(), train["target"])
		tstresult = percentError(trainer.testOnClassData(dataset=test), test["target"])

		# Calculate current training and test set accuracy (not error!)
		pred = net.activateOnDataset(test)
		preds = [y.argmax() for y in pred]
		test_acc = accuracy_score(preds, testy, normalize=True)
		pred = net.activateOnDataset(train)
		preds = [y.argmax() for y in pred]
		train_acc = accuracy_score(preds, trainy, normalize=True)
		print("epoch: {}, train accuracy: {}, test accuracy: {}".format(trainer.totalepochs, train_acc, test_acc))

	return train_acc, test_acc

@timeit
def ann_learning_curve(trainx, trainy, testx, testy, n_hidden=[5, 3],
						n_iter=5, cv=5, train_sizes=np.linspace(.1, 1.0, 10)):
	""" Returns the learning curve for artificial neural networks, i.e. the
	training and test accuracies (not error!). The input variables are:
	trainx - 		features of the training data
	trainy - 		labels of the training data
	testx - 		features of the test data
	testy - 		labels of the test data
	n_hidden - 		2-element list of hidden layer sizes
	n_iter - 		# of epochs to stop training at
	cv - 			the number of trainings to be average for more accurate estimates
	train_sizes -	list of training size proportions, from (0.0, 1.0] corresponding 
					to 0% to 100% of the full training set set size

	The return variables are:
	train_sizes - 			list of the training set (proportional) sizes, i.e. x axis
	average_train_scores - 	the average training accuracy at each training set size
	average_test_scores - 	the average test accuracy at each training set size
	"""

	cv_train_scores = [[0] * len(train_sizes)]
	cv_test_scores = [[0] * len(train_sizes)]
	for c in range(cv):
		train_scores = []
		test_scores = []
		for ts in train_sizes:
			n_examples = int(round(len(trainx) * ts))
			rows = random.sample(range(len(trainx)), n_examples)
			subx = trainx.iloc[rows, ]
			suby = trainy.iloc[rows, ]
			start = time.time()
			a, b = test_ann2(subx, suby, testx, testy,
								n_hidden=n_hidden, n_iter=n_iter)
			print("training time: {} secs".format(time.time() - start))
			current_train_score = a
			current_test_score = b
			train_scores.append(current_train_score)
			test_scores.append(current_test_score)
		cv_train_scores.append(train_scores)
		cv_test_scores.append(test_scores)
	average_train_scores = [sum(i) / cv for i in zip(*cv_train_scores)]
	average_test_scores = [sum(i) / cv for i in zip(*cv_test_scores)]
	return train_sizes, average_train_scores, average_test_scores

def run_ann(train_sizes=np.linspace(0.1, 1, 10)):
	ts = train_sizes


	#Clinvar Data
	tx, ty, vx, vy = load_data("./dataset/clinvar/processed_2.csv")
	a, b, c = ann_learning_curve(tx, ty, vx, vy, n_hidden=[5, 3], n_iter=10,
									cv=5, train_sizes=ts)
	results = [a, b, c]
	with open("./results/clinvar/ann_results.pickle", "wb") as f:
		pickle.dump(results, f)

	"""
	#Medical Cost Data
	tx, ty, vx, vy = load_data_medicalcost("./dataset/medicalcost/processed_medicalcost.csv")
	a, b, c = ann_learning_curve(tx, ty, vx, vy, n_hidden=[5, 3], n_iter=5,
									cv=5, train_sizes=ts)
	results = [a, b, c]
	with open("./results/medicalcost/ann_results_mc.pickle", "wb") as f:
		pickle.dump(results, f)
	"""
def main():
	#trainx, trainy, testx, testy = load_data("./dataset/clinvar/processed_2.csv")
	#trainx, trainy, testx, testy = load_data_medicalcost("./dataset/medicalcost/processed_medicalcost.csv")
	#test_ann(trainx, trainy, testx, testy)
	#test_ann2(trainx, trainy, testx, testy)
	#run_ann()
	plot_learning_curves("./results/clinvar/ann_results.pickle", "Clinvar Data - Neural Networks")
	#plot_learning_curves("./results/medicalcost/ann_results_mc.pickle", "Medical Cost Data - Neural Networks")

if __name__ == "__main__":
	main()