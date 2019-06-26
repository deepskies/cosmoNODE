import scipy as sci
import pandas as pd
import numpy as np

import torch
import torchvision as tv


if __name__ == '__main__':
	df = pd.read_csv('./data/test_set_sample.csv')
	objects = [obj for obj in df.groupby('object_id')]
	print(len(objects))
	print(df.describe())
	