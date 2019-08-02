# simple macros so i can make KLEEN code
import pandas as pd
import numpy as np

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


def split_rand(length, test_frac):
    # given int (representing timeseries length) and fraction to sample
    # returns np array of ints corresponding to the indices of the data
    # i'm not passing the data itself to this because i imagine that it would be slower

    indices, split_idx = split_index(length, test_frac)

    cp = indices.copy()
    np.random.shuffle(cp)
    test_indices = cp[1:split_idx]  # need t0 in train
    train_indices = np.delete(indices, test_indices)
    return train_indices, test_indices


def split_cutoff(length, test_frac):
    train_frac = 1 - test_frac
    indices, split_idx = split_index(length, train_frac)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return train_indices, test_indices


def split_index(length, test_frac):
    indices = np.arange(length)
    split_idx = round(test_frac * length)
    return indices, split_idx
