import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

import demo as d
import macros as m


class LSST(Dataset):
	def __init__(self):
		self.demo = d.Demo()
		self.demo_len = len(df)

		self.items = self.demo.merged_objs

		self.

	def __getitem__(self, index):
		# index is object_id

		obj = self.items[index]
		target = obj.target[0]

		# haven't rigorously checked that there aren't other columns that are linearly
		# dependent with the target value
		
		obj = obj.drop(['object_id', 'target'], axis=1)
		
		return (obj, target)

	def __len__(self):
		return self.df_len

	def join(self):
		# given obj id, gets light curve data and meta data
		pass
