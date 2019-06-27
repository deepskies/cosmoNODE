# simple macros so i can make KLEEN code
import pandas as pd
import torch

from sklearn import preprocessing

ID = 'object_id'

def read_multi(fns):
	dfs = []
	for fn in fns:
		df = pd.read_csv('./data/' + fn + '.csv')
		dfs.append(df)
	return dfs

def scale_df(df):
	x = df.values  # returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df = pd.DataFrame(x_scaled, columns=df.columns)
	return df