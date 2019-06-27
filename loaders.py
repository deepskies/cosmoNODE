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
		self.t_items = [torch.tensor(elt.values) for elt in self.items]
		
		self.padded_items = torch.nn.utils.rnn.pad_sequence(self.t_items, batch_first=True)

		self.train_len = len(self.items)

	def __getitem__(self, index):
		# index is object_id

		obj = self.padded_items[index]
		target = obj.target.iloc[0]

		# haven't rigorously checked that there aren't other columns that are linearly
		# dependent with the target value

		obj = obj.drop(['object_id', 'target'], axis=1)
		
		return (obj, target)

	def __len__(self):
		return self.train_len

	def join(self):
		# given obj id, gets light curve data and meta data
		# TODO
		pass

	def item_pad(self):
		# for item in self.items:
		pass