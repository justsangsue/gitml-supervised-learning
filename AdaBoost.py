from utilities import *

@timeit
def test_adaboost(trainx, trainy, testx, testy, max_depth=6, n_estimators=600,
				learning_rate=1.5, algorithm="SAMME"):
	""" Train and test an AdaBoost classifier using decision trees as a weak learner.
	max_depth is the maximum size of the trees used as weak learners, n_estimators
	is the number learners to combine, learning_rate is self-explanatory, and
	algorithm can be either "SAMME" or "SAMME.R".
	"""
	bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators,
							learning_rate=learning_rate, algorithm=algorithm)
	bdt.fit(trainx, trainy)
	print("max_depth = {}, n_estimators = {}, learning_rate = {}, algorithm = {}".format(
		max_depth, n_estimators, learning_rate, algorithm))
	print("train accuracy: {}".format(bdt.score(trainx, trainy)))
	print("test accuracy: {}".format(bdt.score(testx, testy)))
	return bdt

def run_adaboost(train_sizes=np.linspace(0.1, 1, 10)):
	"""
	#Clinvar Data
	ts = train_sizes
	tx, ty, vx, vy = load_data("./dataset/clinvar/processed_2.csv")
	bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=10),
								n_estimators=20, learning_rate=1.5, algorithm="SAMME")
	a, b, c = get_learning_curve(bdt, tx, ty, vx, vy, cv=5, train_sizes=ts)
	results = [a, b, c]
	with open("./results/clinvar/bdt_results_2.pickle", "wb") as f:
		pickle.dump(results, f)
	"""

	#Medical Cost Data
	ts = train_sizes
	tx, ty, vx, vy = load_data_medicalcost("./dataset/medicalcost/processed_medicalcost.csv")
	bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5),
								n_estimators=400, learning_rate=1.5, algorithm="SAMME")
	a, b, c = get_learning_curve(bdt, tx, ty, vx, vy, cv=5, train_sizes=ts)
	results = [a, b, c]
	with open("./results/medicalcost/bdt_results_mc.pickle", "wb") as f:
		pickle.dump(results, f)

def main():
	#trainx, trainy, testx, testy = load_data("./dataset/clinvar/processed_2.csv")
	#test_adaboost(trainx, trainy, testx, testy, max_depth=1, n_estimators=20)
	run_adaboost()
	#plot_learning_curves("./results/clinvar/bdt_results_2.pickle", "Clinvar Data - Boosted Decision Tree (AdaBoost)")
	plot_learning_curves("./results/medicalcost/bdt_results_mc.pickle", "Medical Cost Data - Boosted Decision Tree (AdaBoost)")
if __name__ == "__main__":
	main()