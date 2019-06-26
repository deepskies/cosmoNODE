# cosmoNODE
Neural Ordinary Differential Equations for Astronomical Sky Surveys

I am using the Kaggle PLAsTiCC-2018 dataset found here: https://www.kaggle.com/c/PLAsTiCC-2018/

The goal is to see if there is any improvement in the classification accuracy using NODEs.

Given that ODEs are commonly used for time series analysis, my hypothesis is that there will be a measureable improvement.

https://colab.research.google.com/drive/1e9g_X_DRhREfIhSVqwXqPqTriWVPRVZc





Time-of documentation:
6/26/19

11:04 trained using anandijain/sip/gym-sip/regression.py neural network on training_set.csv of LSST and tested on the test_set_sample.py
	
	+ the model is just linear 5 -> 20 -> 3 -> 8 -> 4 -> 1 (definitely not optimal, but wanted to just train on anything)

	+ the data was scaled -1 to 1 and the p-value for correct flux prediction was 0.1

	+ the inputs were all of the columns besides the flux label.

	+ this is most likey not proper, as flux_err for an input might not be kosher

	+ the accuracy on the training set was [correct guesses: 9475 / total guesses: 11107] ~ 0.8531 %
