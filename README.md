# cosmoNODE
Neural Ordinary Differential Equations for Astronomical Sky Surveys

I am using the Kaggle PLAsTiCC-2018 dataset found here: https://www.kaggle.com/c/PLAsTiCC-2018/

The goal is to see if there is any improvement in the classification accuracy using NODEs.

Given that ODEs are commonly used for time series analysis, my hypothesis is that there will be a measureable improvement.

https://colab.research.google.com/drive/1e9g_X_DRhREfIhSVqwXqPqTriWVPRVZc

https://colab.research.google.com/drive/1WUNbQV0g-k1wSRueQyuftZx6XKEr5OBx

# TODO priority queue:
	1. build NODE classification demo!!!

	2. write validation nets for testing against NODEs 

	3. documented jupyter notebook for presenting

	4. fix pd.merge memory inefficiency with onsite concat 

	5. base index on 'mjd' column (time series indexing)

	+ 



# Things that seem interesting but later:
	
	1. tf.distribute