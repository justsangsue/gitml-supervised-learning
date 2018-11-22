from utilities import *

@timeit
def test_knn(trainx, trainy, testx, testy, k=14, weights="distance", p=1):
	""" Train and test a K-NN classifier. k is the number of neighbors,
	weights is either "distance" or "uniform" and refers to how the
	K neighbors are weighted, and p is the norm of the distance, i.e. p=2
	gives L2 norm.
	"""
	knn = neighbors.KNeighborsClassifier(k, weights=weights, p=p)
	knn.fit(trainx, trainy)
	print("k = {}, weights = {}, p = {}".format(k, weights, p))
	print("train accuracy: {}".format(knn.score(trainx, trainy)))
	print("test accuracy: {}".format(knn.score(testx, testy)))
	return knn

def run_knn(train_sizes=np.linspace(0.1, 1, 10)):
	"""Run knn with k = 1, 3, 8, 15, 50"""
	
	"""
	#Clinvar Data
	ts = train_sizes
	tx, ty, vx, vy = load_data("./dataset/clinvar/processed_2.csv")
	for k in [1, 3, 8, 15, 50]:
		print("Running " + str(k) + "-NN")
		knn = neighbors.KNeighborsClassifier(k, weights="distance", p=1) 
		a, b, c = get_learning_curve(knn, tx, ty, vx, vy, cv=3, train_sizes=ts)
		results = [a, b, c]
		with open("./results/knn_results_" + str(k) + ".pickle", "wb") as f:
			pickle.dump(results, f)
	"""

	#Medical Data
	ts = train_sizes
	tx, ty, vx, vy = load_data_medicalcost("./dataset/medicalcost/processed_medicalcost.csv")
	for k in [1, 3, 8, 15, 40]:
		print("Running " + str(k) + "-NN")
		knn = neighbors.KNeighborsClassifier(k, weights="distance", p=1) 
		a, b, c = get_learning_curve(knn, tx, ty, vx, vy, cv=10, train_sizes=ts)
		results = [a, b, c]
		with open("./results/medicalcost/knn_results_" + str(k) + "_mc.pickle", "wb") as f:
			pickle.dump(results, f)

def plot_learning_curves(show="clinvar"):

	"""Clinvar Data"""
	if show == "clinvar":
		with open("./results/clinvar/knn_results_1.pickle", "rb") as f:
			knn1 = pickle.load(f)
		with open("./results/clinvar/knn_results_3.pickle", "rb") as f:
			knn3 = pickle.load(f)
		with open("./results/clinvar/knn_results_8.pickle", "rb") as f:
			knn8 = pickle.load(f)
		with open("./results/clinvar/knn_results_15.pickle", "rb") as f:
			knn15 = pickle.load(f)
		with open("./results/clinvar/knn_results_50.pickle", "rb") as f:
			knn50 = pickle.load(f)
		fig = plt.figure()
		ax = plt.subplot(111)
		#ax.plot(knn1[0], knn1[1], "r--", knn1[0], knn1[2], "r-")
		ax.plot(knn1[0], knn1[1], "r-", knn1[0], knn1[2], "r--",
				knn3[0], knn3[1], "b-", knn3[0], knn3[2], "b--",
				knn8[0], knn8[1], "m-", knn8[0], knn8[2], "m--",
				knn15[0], knn15[1], "g-", knn15[0], knn15[2], "g--",
				knn50[0], knn50[1], "c-", knn50[0], knn50[2], "c--")
		plt.xlabel("Fractional Train Set Size")
		plt.ylabel("Accuracy (out of 1)")
		plt.title("Clinvar Data - kNNs")
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
		ax.legend(["1-NN Train Accuracy",
					"1-NN Test Accuracy",
					"3-NN Train Accuracy",
					"3-NN Test Accuracy",
					"8-NN Train Accuracy",
					"8-NN Test Accuracy",
					"15-NN Train Accuracy",
					"15-NN Test Accuracy",
					"50-NN Train Accuracy",
					"50-NN Test Accuracy"],
					loc="center left", bbox_to_anchor=(1, 0.5),
					ncol=1, fancybox=True, shadow=True,
					prop=FontProperties().set_size("small"))
		plt.show()

	elif show == "medicalcost":
		with open("./results/medicalcost/knn_results_1_mc.pickle", "rb") as f:
			knn1 = pickle.load(f)
		with open("./results/medicalcost/knn_results_3_mc.pickle", "rb") as f:
			knn3 = pickle.load(f)
		with open("./results/medicalcost/knn_results_8_mc.pickle", "rb") as f:
			knn8 = pickle.load(f)
		with open("./results/medicalcost/knn_results_15_mc.pickle", "rb") as f:
			knn15 = pickle.load(f)
		with open("./results/medicalcost/knn_results_40_mc.pickle", "rb") as f:
			knn40 = pickle.load(f)
		fig = plt.figure()
		ax = plt.subplot(111)
		#ax.plot(knn1[0], knn1[1], "r--", knn1[0], knn1[2], "r-")
		ax.plot(knn1[0], knn1[1], "r-", knn1[0], knn1[2], "r--",
				knn3[0], knn3[1], "b-", knn3[0], knn3[2], "b--",
				knn8[0], knn8[1], "m-", knn8[0], knn8[2], "m--",
				knn15[0], knn15[1], "g-", knn15[0], knn15[2], "g--",
				knn40[0], knn40[1], "c-", knn40[0], knn40[2], "c--")
		plt.xlabel("Fractional Train Set Size")
		plt.ylabel("Accuracy (out of 1)")
		plt.title("Medical Cost Data - kNNs")
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
		ax.legend(["1-NN Train Accuracy",
					"1-NN Test Accuracy",
					"3-NN Train Accuracy",
					"3-NN Test Accuracy",
					"8-NN Train Accuracy",
					"8-NN Test Accuracy",
					"15-NN Train Accuracy",
					"15-NN Test Accuracy",
					"40-NN Train Accuracy",
					"40-NN Test Accuracy"],
					loc="center left", bbox_to_anchor=(1, 0.5),
					ncol=1, fancybox=True, shadow=True,
					prop=FontProperties().set_size("small"))
		plt.show()

def main():
	#trainx, trainy, testx, testy = load_data("./dataset/clinvar/processed_2.csv")
	#test_knn(trainx, trainy, testx, testy)
	run_knn()
	#plot_learning_curves()
	plot_learning_curves(show="medicalcost")

if __name__ == "__main__":
	main()