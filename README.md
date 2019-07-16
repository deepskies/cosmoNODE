# cosmoNODE
Neural Ordinary Differential Equations for Astronomical Sky Surveys

I am using [Kaggle PLAsTiCC-2018](https://www.kaggle.com/c/PLAsTiCC-2018/)

The goal is to see if there is any improvement in the classification accuracy using NODEs.

Given that ODEs are commonly used for time series analysis,
my hypothesis is that there will be a measurable improvement.

# [Project Overview document](https://drive.google.com/open?id=1dDKOfZrUGG_9MTxTWis1rhZ4L-IAFqjq8vEfahfMiVs)

# [Overleaf Paper](https://www.overleaf.com/read/pznqtfcgzxyp):
	+ For edit access ask Anand (anandj @ uchicago dot edu)

# [Colab Notebooks](https://drive.google.com/open?id=1twyeXpB2EeFEyGj7Y61C9KN7vSuHcUv0):

# Stable tests:
	+ keras_classify.py

	+ (semi) simple_torch.py

	+ ode_demo.py

# Repository Overview
Neural differential equations have not been applied to many fields of ML.
This repository includes astronomical applications of neural differentials.
We implement baseline implementations for classification to cross validate
the NODE and ANODE algorithms.


## requirements:
	- torchdiffeq by Chen et all (2018)
	- augmented neural ODEs
	- stuff in requirements.txt
	- [Kaggle PLAsTiCC-2018 data](https://www.kaggle.com/c/PLAsTiCC-2018/data)
		- place this in ./demos/data/
