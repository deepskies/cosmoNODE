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
			 5 : 'y',} 

df = pd.read_csv('./data/test_set_sample.csv')
objects = [obj for obj in df.groupby(by='object_id', as_index=False)] 
print(len(objects))
print(df.describe())

obj = objects[0]

obj = obj[1]  # tuple -> df
colors = []
bands = obj['passband']

for band in bands:
	colors.append(passbands[band])

plt_x = obj['mjd']
plt_y = obj['flux']
# want to color dots, according to passband
# going thru every row is most likely inefficient.

plt.scatter(plt_x, plt_y, c=colors, s=1)
plt.show()
# type(objects[0]) == tuple
