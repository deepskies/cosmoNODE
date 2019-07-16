# simple macros so i can make KLEEN code
import pandas as pd
import torch

from sklearn import preprocessing

ID = 'object_id'
DATA = './demos/data/'  # location of the data folder wrt the root directory

band_color_map = {0 : 'r',
				 1 : 'g',
				 2 : 'b',
				 3 : 'c',
				 4 : 'm',
				 5 : 'y'}

def read_multi(fns, fillna=False):
	dfs = []
	for fn in fns:
		df = pd.read_csv(DATA + fn + '.csv')
		if fillna:
			df = df.fillna(0)
		dfs.append(df)
	return dfs

def scale_df(df):
	x = df.values  # returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df = pd.DataFrame(x_scaled, columns=df.columns)
	return df