from utilities import *

@timeit
def test_svm(trainx, trainy, testx, testy, C=1):
	""" Train and test a SVM classifier. kernel can be either 
	"poly", "linear", "rbf", or "sigmoid"."""
	sv = svm.LinearSVC(C=C)
	sv.fit(trainx, trainy) #Time consuming step
	print("train accuracy: {}".format(sv.score(trainx, trainy)))
	print("test accuracy: {}".format(sv.score(testx, testy)))
	return sv

def run_svm(train_sizes=np.linspace(0.1, 1, 10)):
	ts = train_sizes


	#Clinvar Data
	tx, ty, vx, vy = load_data("./dataset/clinvar/processed_2.csv")

	sv_linear = svm.LinearSVC(C=1)
	a, b, c = get_learning_curve(sv_linear, tx, ty, vx, vy, cv=10, train_sizes=ts)
	results = [a, b, c]
	with open("./results/clinvar/svm_results_linear.pickle", "wb") as f:
		pickle.dump(results, f)

	n_estimators = 10
	sv_sigmoid = svm.SVC(kernel="sigmoid")
	a, b, c = get_learning_curve(sv_sigmoid, tx, ty, vx, vy, cv=1, train_sizes=ts)
	results = [a, b, c]
	with open("./results/clinvar/svm_results_sigmoid.pickle", "wb") as f:
		pickle.dump(results, f)
	"""
	n_estimators = 10
	sv_poly = svm.SVC(kernel="poly")
	a, b, c = get_learning_curve(sv_poly, tx, ty, vx, vy, cv=1, train_sizes=ts)
	results = [a, b, c]
	with open("./results/clinvar/svm_results_poly.pickle", "wb") as f:
		pickle.dump(results, f)
	"""

	#Medical Cost Data
	load_data_medicalcost("./dataset/medicalcost/processed_medicalcost.csv")

	sv_linear = svm.LinearSVC(C=1)
	a, b, c = get_learning_curve(sv_linear, tx, ty, vx, vy, cv=10, train_sizes=ts)
	results = [a, b, c]
	with open("./results/medicalcost/svm_results_linear_mc.pickle", "wb") as f:
		pickle.dump(results, f)

	n_estimators = 10
	sv_sigmoid = svm.SVC(kernel="sigmoid")
	a, b, c = get_learning_curve(sv_sigmoid, tx, ty, vx, vy, cv=5, train_sizes=ts)
	results = [a, b, c]
	with open("./results/medicalcost/svm_results_sigmoid_mc.pickle", "wb") as f:
		pickle.dump(results, f)

	"""
	n_estimators = 10
	sv_poly = svm.SVC(kernel="poly")
	a, b, c = get_learning_curve(sv_poly, tx, ty, vx, vy, cv=1, train_sizes=ts)
	results = [a, b, c]
	with open("./results/medicalcost/svm_results_poly_mc.pickle", "wb") as f:
		pickle.dump(results, f)
	"""

def plot_learning_curves(show="clinvar"):
	
	#Clinvar Data
	if show == "clinvar":
		with open("./results/clinvar/svm_results_linear.pickle", "rb") as f:
			sv1 = pickle.load(f)
		with open("./results/clinvar/svm_results_sigmoid.pickle", "rb") as f:
			sv2 = pickle.load(f)
		with open("./results/clinvar/svm_results_poly.pickle", "rb") as f:
			sv3 = pickle.load(f) 

		fig = plt.figure()
		ax = plt.subplot(111)
		plt.plot(sv1[0], sv1[1], "r-", sv1[0], sv1[2], "r--",
				sv2[0], sv2[1], "b-", sv2[0], sv2[2], "b--",
				sv3[0], sv3[1], "g-", sv3[0], sv3[2], "g--")
		plt.xlabel("Fractional Training Set Size")
		plt.ylabel("Accuracy (out of 1)")
		plt.title("Clinvar Data - SVMs")
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width, box.height])
		plt.legend(["Linear Kernel SVM Train",
					"Linear Kernel SVM Test",
					"Sigmoid Kernel SVM Train",
					"Sigmoid Kernel SVM Test",
					"Polynomial Kernel SVM Train",
					"Polynomial Kernel SVM Test"],
					loc="center left", bbox_to_anchor=(1, 0.5),
					ncol=1, fancybox=True, shadow=True, 
					prop = FontProperties().set_size("small"))
		plt.show()

	"""Medical Cost Data"""
	if show == "medicalcost":
		with open("./results/medicalcost/svm_results_linear_mc.pickle", "rb") as f:
			sv1 = pickle.load(f)
		with open("./results/medicalcost/svm_results_sigmoid_mc.pickle", "rb") as f:
			sv2 = pickle.load(f)
		with open("./results/medicalcost/svm_results_poly_mc.pickle", "rb") as f:
			sv3 = pickle.load(f) 

		fig = plt.figure()
		ax = plt.subplot(111)
		plt.plot(sv1[0], sv1[1], "r-", sv1[0], sv1[2], "r--",
				sv2[0], sv2[1], "b-", sv2[0], sv2[2], "b--",
				sv3[0], sv3[1], "g-", sv3[0], sv3[2], "g--")
		plt.xlabel("Fractional Training Set Size")
		plt.ylabel("Accuracy (out of 1)")
		plt.title("Medical Cost Data - SVMs")
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width, box.height])
		plt.legend(["Linear Kernel SVM Train",
					"Linear Kernel SVM Test",
					"Sigmoid Kernel SVM Train",
					"Sigmoid Kernel SVM Test",
					"Polynomial Kernel SVM Train",
					"Polynomial Kernel SVM Test"],
					loc="center left", bbox_to_anchor=(1, 0.5),
					ncol=1, fancybox=True, shadow=True, 
					prop = FontProperties().set_size("small"))
		plt.show()
def main():
	#trainx, trainy, testx, testy = load_data("./dataset/clinvar/processed_2.csv")
	#test_svm(trainx, trainy, testx, testy)
	#run_svm()
	plot_learning_curves()
	plot_learning_curves(show="medicalcost")
if __name__ == "__main__":
	main()