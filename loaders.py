import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


import demo as d
import macros as m

# pytorch loader
class LSST(Dataset):
	def __init__(self, data_class=d.Quick()):
		
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

class FluxLoader(Dataset):
	def __init__(self):

		full_df = pd.read_csv('./data/training_set.csv')
		
		self.split_pct = 0.7

		
		self.df = full_df[['object_id', 'mjd', 'flux']]
		
		self.items = self.df.groupby(by='object_id', as_index=False)
		self.items = [item[1].drop(m.ID, axis=1) for item in self.items]

		self.item = self.items[0]
		# print(self.item)

		self.t_items = [torch.tensor(item.values) for item in self.items]

		self.padded_items = torch.nn.utils.rnn.pad_sequence(self.t_items, batch_first=True)


		# print(self.padded_items[0])

		self.train_len = len(self.items)
	
	def __getitem__(self, index):
		obj = self.padded_items[index]
		times = obj[:, 0]
		fluxes = obj[:, 1]
		return (times, fluxes)  # (t, y)

	def __len__(self):

		pass

if __name__ == '__main__':
	f = FluxLoader()


