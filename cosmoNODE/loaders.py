import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import torch
from torch.utils.data import Dataset

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from cosmoNODE import macros as m

'''
Anode Dataset was written 7/15/19 and is probably the best one so far,
but there are things that need to be added and fixed.
For one, I think that it could be more malleable to channels, treating passbands
as channels.

'''
class Anode(Dataset):
	def __init__(self, df_cols=['mjd', 'flux']):
			fns = ['training_set', 'training_set_metadata']
			self.raw, self.raw_meta = m.read_multi(fns, fillna=True)

			self.df = self.raw[[m.ID] + df_cols]
			self.df_meta = self.raw_meta[['object_id', 'target']].sort_values(by=m.ID)
			val_counts = self.df['object_id'].value_counts()
			self.seq_max_len = val_counts.max()
			self.input_dim = len(df_cols)

			self.classes = self.df_meta['target'].unique()
			# print(f'classes: {self.classes}, type {type(self.classes)}')
			self.classes.sort()
			# print(f'sorted: {self.classes}')
			self.class_list = self.classes.tolist()

			self.id_group = self.df.groupby(by=m.ID, as_index=False)
			self.objs = [elt for elt in self.id_group]

			self.obj_count = len(self.objs)
			self.tups = []
			self.create_tuples()

	def class_to_tensor(self, target):
		class_index = self.class_list.index(target)
		target_tensor = torch.zeros(14, dtype=torch.double)
		target_tensor[class_index] = 1
		return target_tensor

	def pad(self, obj_data):
		obj_len = len(obj_data)
		cat_len = self.seq_max_len - obj_len

		padding = torch.zeros([cat_len, self.input_dim], dtype=torch.double)

		without_id = obj_data.drop('object_id', axis=1)
		obj_data_tensor = torch.tensor(without_id.values, dtype=torch.double)
		x_tensor = torch.cat([obj_data_tensor, padding])
		return x_tensor

	def __getitem__(self, index):
		x, y = self.tups[index]
		x = x.reshape(1, -1)
		return (x, y)

	def __len__(self):
		return self.obj_count

	def create_tuples(self):
		# redo this to not loop but use a map or lambda
		for obj in self.objs:

			obj_id, obj_data = obj

			x_tensor = self.pad(obj_data)

			obj_meta = self.df_meta[self.df_meta['object_id'] == obj_id]
			obj_target_class = obj_meta['target'].iloc[0]

			y_tensor = self.class_to_tensor(obj_target_class)
			self.tups.append((x_tensor, y_tensor))



'''
What shape should __getitem__ return?
Returning a single line seems inefficient. Fix later
For now im batching in __getitem__
'''

class NDim(Dataset):
	def __init__(self, batch_size=16):

		self.batch_size = batch_size

		fns = ['training_set', 'training_set_metadata']
		self.tr, self.tr_meta = m.read_multi(fns)

		self.raw = pd.merge(self.tr, self.tr_meta, on='object_id')
		self.raw = self.raw.fillna(0)

		# is it hacking to give the model obj_id?
		self.obj_ids = self.raw['object_id']

		self.df = self.raw.drop(['object_id', 'target'], axis=1)

		self.t = self.df['mjd']  # 1D list of values to calculate Y for in ODE
		self.y = self.df.drop('mjd', axis=1)

		self.y_dim = len(self.y.columns)

		self.train_len = len(self.df)

	def __getitem__(self, index):
		try:
			np_t = self.t.iloc[index:index + self.batch_size].values
			np_y = self.y.iloc[index:index + self.batch_size].values

			batch_t = torch.tensor(np_t, dtype=torch.double).reshape(-1, 1)
			batch_y = torch.tensor(np_y, dtype=torch.double)
			item = (batch_t, batch_y)
		except IndexError:
			item = (None, None)

		return item

	def __len__(self):
	    return self.train_len


class Quick:
	def __init__(self, cols=[m.ID, 'mjd', 'flux']):
		# doesn't read the absurdly large test set sample

		self.fns = ['training_set', 'training_set_metadata']
		self.df, self.meta_df = m.read_multi(self.fns)
		self.raw = self.df

		self.df = self.df[cols]

		self.df_grouped = self.df.groupby(by=m.ID, as_index=False)

		# self.target_classes = self.meta_df['target'].unique()
		# self.target_classes.sort()

		# self.class_list = self.target_classes.tolist()

		# self.merged = pd.merge(self.df, self.meta_df, on=m.ID)
		# self.merged= self.merged.fillna(0).astype(np.float32)

		# self.grouped = self.merged.groupby(by=m.ID, as_index=False)

		# self.grouped = self.merged.groupby(by=[m.ID, 'passband'], as_index=False)

		self.unscaled_objs = [obj[1] for obj in self.df_grouped]

	def graph_test(self):
		graph_object(self.unscaled_objs, 234)


class LSST(Dataset):
	def __init__(self, data_class=Quick(cols=[m.ID, 'mjd', 'flux'])):

		self.data_class = data_class

		# self.pad_len = self.data_class.seq_max_len

		# self.items = self.data_class.merged_objs
		self.items = self.data_class.unscaled_objs

		# inefficient, want to create label list earlier
		self.labels = [elt.target.iloc[0] for elt in self.items]

		self.raw_items = [item.drop(['object_id', 'target'], axis=1) for item in self.items]

		self.t_items = [torch.tensor(elt.values) for elt in self.raw_items]
		self.padded_items = torch.nn.utils.rnn.pad_sequence(self.t_items, batch_first=True)

		self.class_list = self.data_class.class_list

		self.train_len = len(self.items)
		self.item = self.__getitem__(0)

		self.input_shape = self.item[0].shape
		self.output_shape = len(self.data_class.class_list)  # type() == torch.Size(14)
		self.tuples = []

		print('torch LSST Dataset initialized\n')

	def __getitem__(self, index):
		# index is object_id

		obj = self.padded_items[index]
		target = self.labels[index]  # this is just the number of the class, we want a 14

		class_index = self.class_list.index(target)

		target_tensor = torch.zeros(14)
		target_tensor[class_index] = 1


		# haven't rigorously checked that there aren't other columns that are linearly
		# dependent with the target value

		return (obj, target_tensor)

	def __len__(self):
		return self.train_len

	def join(self):
		# given obj id, gets light curve data and meta data
		# TODO
		# meta = self.data_class.

		pass
	def make_tuples(self):
		for obj in self.data_class.df_grouped:
			obj_id = obj[0]
			obj_data = obj[1]
			obj_target = self.data_class.meta_df[self.data_class.meta_df['target' == obj_id]]
			self.tuples.append((obj_data, obj_target))


class FluxLoader(Dataset):
	def __init__(self, fn='single_obj'):

		full_df = pd.read_csv('./demos/data/' + fn + '.csv')

		self.split_pct = 0.7

		self.df = full_df[['object_id', 'mjd', 'flux']]

		self.items = self.df.groupby(by='object_id', as_index=False)
		self.items = [item[1].drop(m.ID, axis=1).sort_values(by='mjd') for item in self.items]

		self.item = self.items[0]
		# print(self.item)

		self.t_items = [torch.tensor(item.values) for item in self.items]

		# self.padded_items = torch.nn.utils.rnn.pad_sequence(self.t_items, batch_first=True)
		# print(self.padded_items[0])

		self.train_len = len(self.items)

	def __getitem__(self, index):
		# obj = self.padded_items[index]
		obj = self.t_items[index]
		times = obj[:, 0]
		fluxes = obj[:, 1]
		return (times, fluxes)  # (t, y)

	def __len__(self):
		return self.train_len


class DataPrep:
	def __init__(self):
		self.fns = ['training_set', 'training_set_metadata', 'test_set_sample', 'test_set_metadata']

		self.df, self.meta_df, self.test_df, self.test_meta = m.read_multi(self.fns)

		self.set_list = [self.df, self.meta_df, self.test_df, self.test_meta]

		# fill in Na

		self.tr_objs = [obj for obj in self.df.groupby(by=m.ID, as_index=False)]
		self.te_objs = [obj for obj in self.df.groupby(by=m.ID, as_index=False)]

		self.tr_objs_pb = [obj for obj in self.df.groupby(by=[m.ID, 'passband'], as_index=False)]
		self.te_objs_pb = [obj for obj in self.df.groupby(by=[m.ID, 'passband'], as_index=False)]


		self.seq_max_len = self.df[m.ID].value_counts().max()

		'''
		was asserting that df1.size + df2.size == merged
		this is not the case, and theres isnt some cool compression used to prevent
		duplicating metadata
		pd.merge will be fine for now
		'''

		self.merged = pd.merge(self.df, self.meta_df, on=m.ID)
		self.test_merged = pd.merge(self.test_df, self.test_meta, on=m.ID)

		self.merged= self.merged.fillna(0).astype(np.float32)
		self.test_merged= self.test_merged.fillna(0).astype(np.float32)

		self.grouped = self.merged.groupby(by=m.ID, as_index=False)
		self.test_grouped = self.test_merged.groupby(by=m.ID, as_index=False)

		self.unscaled_objs = [obj[1] for obj in self.grouped]
		self.test_unscaled_obj = [obj[1] for obj in self.test_grouped]

		self.merged = m.scale_df(self.merged)
		self.test_merged = m.scale_df(self.test_merged)

		self.merged_objs = [obj[1] for obj in self.grouped]
		self.test_merged_objs = [obj[1] for obj in self.test_grouped]

		# self.merged_pbs = [obj[1] for ]

		self.input_size = len(self.merged.columns) - 2  # -2 for the obj id and target

		self.target_classes = self.meta_df['target'].unique()
		self.target_classes.sort()

		self.class_list = self.target_classes.tolist()

		self.output_size = len(self.target_classes)

		print('demo initialized\n')

	def lookup(self, obj_id):
		meta_data = self.meta_df.loc[self.meta_df['object_id'] == obj_id]
		return meta_data


def graph_object(df_list, index, passband=None, df=1):

	obj = df_list[index]

	# obj = obj[1]  # tuple -> df

	if passband is None:
		# use all bands in graph
		pass
	else:
		# passband is type list or None
		obj = obj.loc[obj['passband'].isin(passband)]  # use only data from a given band


	bands = obj['passband']

	# TODO, port to pandas transforms/mapping for efficiency

	colors = []
	for band in bands:
		colors.append(m.band_color_map[band])

	plt_x = obj['mjd']
	plt_y = obj['flux']

	plt.scatter(plt_x, plt_y, c=colors, s=5)
	plt.show()




if __name__ == '__main__':
	f = FluxLoader()
