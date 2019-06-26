import scipy as sci
import pandas as pd
import numpy as np

# import torch
# import torchvision as tv

import matplotlib.pyplot as plt


passbands = {0 : 'r',
			 1 : 'g',
			 2 : 'b',
			 3 : 'c',
			 4 : 'm',
			 5 : 'y'} 


class Demo:
	def __init__(self):
		self.fns = ['training_set', 'training_set_metadata', 'test_set_sample'] 
		
		self.df, self.meta_df, self.test_df = read_multi(self.fns)

		self.test_objs = [obj for obj in self.df.groupby(by='object_id', as_index=False)] 


	def graph_object(self, index=0):
		obj = self.obj[index]
		obj = obj[1]  # tuple -> df
		colors = []
		bands = obj['passband']

		for band in bands:
			colors.append(passbands[band])

		plt_x = obj['mjd']
		plt_y = obj['flux']
		# want to color dots, according to passband
		# going thru every row is most likely inefficient.
		plt.scatter(plt_x, plt_y, c=colors, s=5)
		plt.show()

	def obj_data(self, index=0):
		return self.objects[index]


def read_multi(fns):
	dfs = []
	for fn in fns:
		dfs.append(pd.read_csv('./data/' + fn + '.csv'))
	return dfs