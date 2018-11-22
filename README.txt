To be able to run this code you will need Python 3.6 to be installed.

All code assumes that the working directory is this file, README.txt resides. The structure of the folder is assumed to be as follows.

All code is in the folder called "jwang3013".

  jwang2013
  │   utilities.py
  |   ClinvarDataProcessing.py
  |   MedicalCostDataProcessing.py
  |   DecisionTree.py
  |   AdaBoost.py
  |   kNN.py
  |   SVM.py
  |   NeuralNetworks.py
  │   README.txt
  │   jwang3013-analysis.pdf


The following Python packages are also required:

*  csv
*  numpy
*  pandas
*  pickle
*  pylab
*  functools
*  matplotlib
*  pybrain
*  sklearn  


To repeat the experiment, download the dataset first from kaggle:
Genetic Variant: https://www.kaggle.com/kevinarvai/clinvar-conflicting
Medical Charges: https://www.kaggle.com/mirichoi0218/insurance

Then run data processing:
"ClinvarDataProcessing.py" and "MedicalCostDataProcessing.py"

For each learning algorithm, run the following .py files to train and generate figures
DecisionTree.py
AdaBoost.py
kNN.py
SVM.py
NeuralNetworks.py



