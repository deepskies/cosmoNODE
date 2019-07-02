import scipy as sci
import pandas as pd
import numpy as np

from sklearn import preprocessing

import tensorflow as tf

import matplotlib.pyplot as plt

import macros as m

band_color_map = {0 : 'r',
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

	def graph_object(self, index, passband=None, df=1):

		if df:
			obj = self.tr_objs[index]
		else:
			obj = self.te_objs[index]

		obj = obj[1]  # tuple -> df

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
			colors.append(band_color_map[band])


		plt_x = obj['mjd']
		plt_y = obj['flux']

		plt.scatter(plt_x, plt_y, c=colors, s=5)
		plt.show()

	def lookup(self, obj_id):
		meta_data = self.meta_df.loc[self.meta_df['object_id'] == obj_id]
		return meta_data


class Quick:
	def __init__(self):
		# doesn't read the absurdly large test set sample

		self.fns = ['training_set', 'training_set_metadata']
		self.df, self.meta_df = m.read_multi(self.fns)

		self.df_grouped = self.df.groupby(by=m.ID, as_index=False)

		self.target_classes = self.meta_df['target'].unique()
		self.target_classes.sort()

		self.class_list = self.target_classes.tolist()

		self.merged = pd.merge(self.df, self.meta_df, on=m.ID)
		self.merged= self.merged.fillna(0).astype(np.float32)

		self.grouped = self.merged.groupby(by=m.ID, as_index=False)

		# self.grouped = self.merged.groupby(by=[m.ID, 'passband'], as_index=False)

		self.unscaled_objs = [obj[1] for obj in self.grouped]  	

