import scipy as sci
import pandas as pd
import numpy as np

import tensorflow as tf

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
		
		self.set_list = [self.df, self.meta_df, self.test_df, self.test_meta]

		# fill in Na

		self.tr_objs = [obj for obj in self.df.groupby(by=m.ID, as_index=False)] 
		self.te_objs = [obj for obj in self.df.groupby(by=m.ID, as_index=False)] 

		self.seq_max_len = self.df[m.ID].value_counts().max()

		'''
		was asserting that df1.size + df2.size == merged 
		this is not the case, and theres isnt some cool compression used to prevent
		duplicating metadata 
		pd.merge will be fine for now
		'''

		self.merged = pd.merge(self.df, self.meta_df, on=m.ID)
		self.test_merged = pd.merge(self.test_df, self.test_meta, on=m.ID)
		self.merged= pd.to_numeric(self.merged, errors='coerce').fillna(0).astype(np.int64)
		self.test_merged= pd.to_numeric(self.test_merged, errors='coerce').fillna(0).astype(np.int64)

		self.merged_objs = [obj[1] for obj in self.merged.groupby(by=m.ID, as_index=False)] 
		self.test_merged_objs = [obj[1] for obj in self.merged.groupby(by=m.ID, as_index=False)] 

		self.input_size = len(self.merged.columns) - 2  # -2 for the obj id and target
		self.output_size = len(self.merged['target'].unique())

		

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

	def lookup(self, obj_id):
		meta_data = self.meta_df.loc[self.meta_df['object_id'] == obj_id]
		return meta_data





