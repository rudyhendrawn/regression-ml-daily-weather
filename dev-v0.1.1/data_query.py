# Module for querying data from sqlite then convert to dataframe
import os
import pandas as pd
import sqlite3

from dotenv import load_dotenv
from pandas.core.frame import DataFrame

# Create a sqlite connection to sqlite database in db folder
load_dotenv()
db_path = os.getenv("DB_PATH")
# DB_PATH = "db/data.sqlite"

def connect_to_db():
	con = None
	try:
		con = sqlite3.connect(db_path)
		return con
	except sqlite3.Error as e:
		print(e, "Koneksi ke database gagal!")
		if con is not None:
			con.close()

def get_train_data() -> DataFrame: 
	con = connect_to_db()
	df = pd.read_sql_query("SELECT * FROM tb_train", con)
	con.close()
	return df

def get_test_data() -> DataFrame:
	con = connect_to_db()
	df = pd.read_sql_query("SELECT * FROM tb_test", con)
	con.close()
	return df

def get_train_results() -> DataFrame:
	con = connect_to_db()
	df = pd.read_sql_query("SELECT * FROM tb_train_result", con)
	df = rename_column(df)
	con.close()
	return df

def get_test_results() -> DataFrame:
	con = connect_to_db()
	df = pd.read_sql_query("SELECT * FROM tb_test_result", con)
	df = rename_column(df)
	con.close()
	return df

def insert_tb_train(dataframe: DataFrame):
	con = connect_to_db()
	dataframe.to_sql("tb_train", con, if_exists="append", index=False)
	con.close()

def insert_tb_test(dataframe: DataFrame):
	con = connect_to_db()
	dataframe.to_sql("tb_test", con, if_exists="append", index=False)
	con.close()

def insert_into_train_results(dataframe: DataFrame):
	con = connect_to_db()
	dataframe.to_sql("tb_train_result", con, if_exists="append", index=False)
	con.close()

def insert_into_test_results(dataframe: DataFrame):
	con = connect_to_db()
	dataframe.to_sql("tb_test_result", con, if_exists="append", index=False)
	con.close()

def rename_column(dataframe: DataFrame):
	columns = {"date": "Tanggal",
			   "time": "Time",
			   "model": "Model",
			   "jml_atribut": "Jumlah_Atribut",
			   "atribut": "Atribut",
			   "mae": "MAE",
			   "mse": "MSE",
			   "rmse": "RMSE"}
	dataframe.rename(columns=columns, inplace=True)
	return dataframe