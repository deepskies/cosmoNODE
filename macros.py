# simple macros so i can make KLEEN code
import pandas as pd

def read_multi(fns):
	dfs = []
	for fn in fns:
		dfs.append(pd.read_csv('./data/' + fn + '.csv'))
	return dfs

ID = 'object_id'