import scipy as sci
import pandas as pd
import numpy as np

# import torch
# import torchvision as tv

import matplotlib.pyplot as plt

import macros as m

passbands = {0 : 'r',
			 1 : 'g',
			 2 : 'b',
			 3 : 'c',
			 4 : 'm',
			 5 : 'y'} 


class Demo:
	def __init__(self):
		self.fns = ['training_set', 'training_set_metadata', 'test_set_sample', 'test_set_metadata'] 
		
		self.df, self.meta_df, self.test_df, self.test_meta = m.read_multi(self.fns)

		self.tr_objs = [obj for obj in self.df.groupby(by=m.ID, as_index=False)] 
		self.te_objs = [obj for obj in self.df.groupby(by=m.ID, as_index=False)] 

		self.merged = pd.merge(self.df, self.meta_df, on=m.ID)
		self.test_merged = pd.merge(self.test_df, self.test_meta, on=m.ID)

	def graph_object(self, df=0, index=0):

		if df:
			obj = self.tr_objs[index]
		else:
			obj = self.te_objs[index]

		obj = obj[1]  # tuple -> df

		bands = obj['passband']

		# TODO, port to pandas transforms/mapping for efficiency

		colors = []
		for band in bands:
			colors.append(passbands[band])

		plt_x = obj['mjd']
		plt_y = obj['flux']

		plt.scatter(plt_x, plt_y, c=colors, s=5)
		plt.show()

	def obj_data(self, index=0):
		# TODO instant concat with meta
		full_line = self.matchup(obj_id)
		return self.tr_[index]

	def lookup(self, obj_id):
		meta_data = self.meta_df.loc[self.meta_df['object_id'] == obj_id]
		return meta_data



