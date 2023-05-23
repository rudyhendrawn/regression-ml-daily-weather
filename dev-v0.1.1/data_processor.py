# Module for preprocessing the dataset
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import data_query

dataframe = data_query.get_train_data()

# PREPROCESSING DATA
def preprocessed_data(dataframe=dataframe) -> DataFrame:
	# Fill missing values in QFF Column
	dataframe['QFF'].fillna(dataframe['QFF'].mean(), inplace=True)
	return dataframe

def drop_columns(dataframe=dataframe, columns_list=None) -> DataFrame:
	if columns_list is not None:
		used_columns = [col for col in dataframe.columns if col not in columns_list]
		dataframe = dataframe[used_columns]
		return dataframe
	else:
		print("Tidak ada kolom yang dihapus")
		return dataframe

def split_dataset(dataframe=dataframe, test_ratio=0.1) -> DataFrame:
	test_indices = np.random.rand(len(dataframe)) < (test_ratio)
	return dataframe[~test_indices], dataframe[test_indices]

