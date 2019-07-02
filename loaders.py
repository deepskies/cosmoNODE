import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

import demo as d
import macros as m

# pytorch loader
class LSST(Dataset):
	def __init__(self):
		self.demo = d.Demo()

		self.pad_len = self.demo.seq_max_len

		self.items = self.demo.merged_objs
		# inefficient, want to create label list earlier 
		self.labels = [elt.target.iloc[0] for elt in self.items]

		self.raw_items = [item.drop(['object_id', 'target'], axis=1) for item in self.items]

		self.t_items = [torch.tensor(elt.values) for elt in self.raw_items]
		self.padded_items = torch.nn.utils.rnn.pad_sequence(self.t_items, batch_first=True)

		self.train_len = len(self.items)
		self.item = self.__getitem__(0)

		self.input_shape = self.item[0].shape
		self.output_shape = self.demo.output_size  # type() == int

		print('torch LSST Dataset initialized\n')

	def __getitem__(self, index):
		# index is object_id

		obj = self.padded_items[index]
		target = self.labels[index]

		# haven't rigorously checked that there aren't other columns that are linearly
		# dependent with the target value
		
		return (obj, target)

	def __len__(self):
		return self.train_len

	def join(self):
		# given obj id, gets light curve data and meta data
		# TODO
		# meta = self.demo.

		pass

