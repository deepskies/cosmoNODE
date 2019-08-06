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
		self.merged = pd.merge(self.df, self.meta, on='object_id')

		self.mins = self.merged.min()
		self.maxes = self.merged.max()

		self.params = self.merged.columns.drop(['object_id', 'mjd', 'target'])
		self.labels = sorted(self.merged['target'].unique())
		self.num_classes = len(self.labels)

		self.groups = self.merged.groupby('object_id')
		self.curves = []
		self.get_curves()
		self.length = len(self.curves)

	def get_curves(self):
		for i, group in enumerate(self.groups):
			object_id = group[0]
			data = group[1]
			times = torch.tensor(data['mjd'].values)
			values = torch.tensor(data.drop(['mjd', 'target'], axis=1).fillna(0).values)
			mask = torch.ones(values.shape)
			target = data['target'].iloc[0]
			# label = labels.index(data['target'].iloc[0])
			label = one_hot(self.labels, target)
			record = (object_id, times, values, mask, label)
			self.curves.append(record)

	def __getitem__(self, index):
		return self.curves[index]

	def __len__(self):
		# number of light curves in the dataset
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


def variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train",
	data_min = None, data_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][2].shape[1]
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	combined_tt = combined_tt.to(device)

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

	combined_labels = None
	N_labels = 14

	combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
	combined_labels = combined_labels.to(device = device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		tt = tt.to(device)
		vals = vals.to(device)
		mask = mask.to(device)
		if labels is not None:
			labels = labels.to(device)

		indices = inverse_indices[offset:offset + len(tt)]
		offset += len(tt)

		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask

		if labels is not None:
			combined_labels[b] = labels

	combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask,
		att_min = data_min, att_max = data_max)

	if torch.max(combined_tt) != 0.:
		combined_tt = combined_tt / torch.max(combined_tt)

	data_dict = {
		"data": combined_vals,
		"time_steps": combined_tt,
		"mask": combined_mask,
		"labels": combined_labels[:,4]}

	data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
	return data_dict


def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()
