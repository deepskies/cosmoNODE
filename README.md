# cosmoNODE
Neural Ordinary Differential Equations for Astronomical Sky Surveys

I am using the Kaggle PLAsTiCC-2018 dataset found here: https://www.kaggle.com/c/PLAsTiCC-2018/

The goal is to see if there is any improvement in the classification accuracy using NODEs.

Given that ODEs are commonly used for time series analysis, my hypothesis is that there will be a measureable improvement.

https://drive.google.com/open?id=1dDKOfZrUGG_9MTxTWis1rhZ4L-IAFqjq8vEfahfMiVs

# colab notebooks
https://drive.google.com/open?id=1twyeXpB2EeFEyGj7Y61C9KN7vSuHcUv0 


# Stable tests:
	+ keras_classify.py

	+ (semi) simple_torch.py


# TODO priority queue:
	1. build NODE classification demo!!!

	2. write validation nets for testing against NODEs 

		+ train/test splits on train set, since there are no labels on the test set.

		+ chunk by passband to classify each object by individual band

	3. documented jupyter notebook for presenting

	4. fix pd.merge memory inefficiency with onsite concat 

	5. base index on 'mjd' column (time series indexing)


# Things that seem interesting but later:
	
	1. tf.distribute