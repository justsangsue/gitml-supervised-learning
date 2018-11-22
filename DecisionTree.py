from utilities import *

@timeit
def test_decision_tree(trainx, trainy, testx, testy, max_depth=6):
	""" Train and test a decision tree. max_depth is the max depth
	of the tree that can be grown."""
	clf = tree.DecisionTreeClassifier(max_depth=max_depth)
	dt = clf.fit(trainx, trainy)
	print("max_depth = {}".format(max_depth))
	print("train accuracy: {}".format(dt.score(trainx, trainy)))
	print("test accuracy: {}".format(dt.score(testx, testy)))
	return dt

def run_decision_tree(train_sizes=np.linspace(0.1, 1, 10)):
	#Clinvar Data
	ts = train_sizes
	tx, ty, vx, vy = load_data("./dataset/clinvar/processed_2.csv")
	dt = tree.DecisionTreeClassifier(max_depth=50)
	a, b, c = get_learning_curve(dt, tx, ty, vx, vy, cv=5, train_sizes=ts)
	results = [a, b, c]
	with open("./results/clinvar/dt_results.pickle", "wb") as f:
		pickle.dump(results, f)

	"""
	#Medical Cost Data
	ts = train_sizes
	tx, ty, vx, vy = load_data_medicalcost("./dataset/medicalcost/processed_medicalcost.csv")
	dt = tree.DecisionTreeClassifier(max_depth=50)
	a, b, c = get_learning_curve(dt, tx, ty, vx, vy, cv=5, train_sizes=ts)
	results = [a, b, c]
	with open("./results/medicalcost/dt_results_mc.pickle", "wb") as f:
		pickle.dump(results, f)
	"""
def main():
	#trainx, trainy, testx, testy = load_data("./dataset/clinvar/processed_2.csv")
	#test_decision_tree(trainx, trainy, testx, testy, max_depth=6)
	#run_decision_tree()
	plot_learning_curves("./results/clinvar/dt_results.pickle", "Clinvar Data - Decision Tree")
	#plot_learning_curves("./results/medicalcost/dt_results_mc.pickle", "Medical Cost Data - Decision Tree")

if __name__ == "__main__":
	main()
