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


def split(length, test_frac, type='rand'):
	if type == 'rand':
		indices, split_idx = split_index(length, test_frac)
		cp = indices.copy()
		np.random.shuffle(cp)
		test_indices = cp[1:split_idx]  # need t0 in train
		train_indices = np.delete(indices, test_indices)

	elif type == 'cutoff':
		train_frac = 1 - test_frac
		indices, split_idx = split_index(length, train_frac)
		train_indices = indices[:split_idx]
		test_indices = indices[split_idx:]

	else:
		print('unsupported split type')
		return
	train_indices = torch.tensor(train_indices, dtype=torch.long)
	test_indices = torch.tensor(test_indices, dtype=torch.long)
	return train_indices, test_indices


def split_index(length, test_frac):
    indices = np.arange(length)
    split_idx = round(test_frac * length)
    return indices, split_idx


class FluxNet(object):
	def __init__(self):
		self.df = pd.read_csv('./demos/data/training_set.csv')
		self.meta = pd.read_csv('./demos/data/training_set_metadata.csv')
		self.merge = pd.merge(self.df, self.meta, on='object_id')

		self.params = self.merge.columns.drop(['object_id', 'mjd', 'target'])
		self.labels = sorted(self.merge['target'].unique())

		self.groups = self.merge.groupby('object_id')
		self.curves = []
		self.get_curves()
		self.length = len(self.curves)

	def get_curves(self):
		for i, group in enumerate(self.groups):
			object_id = group[0]
			data = group[1]
			times = data['mjd'].values
			values = data.drop(['mjd', 'target'], axis=1).fillna(0).values
			mask = np.ones(values.shape)
			target = data['target'].iloc[0]
			# label = labels.index(data['target'].iloc[0])
			label = one_hot(self.labels, target)
			record = (object_id, times, values, mask, label)
			self.curves.append(record)

	def __getitem__(self, index):
		return self.curves[index]

	def __len__(self):
		return self.length
	#
	# def get_label(self, object_id):
	# 	return self.labels[record_id]


def one_hot(classes, target):
	# classes is list of ints, target is an int in the list
	class_index = classes.index(target)
	target_tensor = torch.zeros(len(classes), dtype=torch.double)
	target_tensor[class_index] = 1
	return target_tensor
