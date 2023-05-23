import os
from lightgbm.sklearn import LGBMRegressor
import numpy as np

import pandas as pd
import lightgbm as lgb
import streamlit as st
import tensorflow as tf
import plotly.express as px
import tensorflow_decision_forests as tfdf

from datetime import datetime
from xgboost import XGBRegressor
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# SET GLOBAL VARIABLES
DATE_COLUMN = 'Date'
FILE_NAME = 'dataset/fklim-ngrai-20112020-final.csv'

@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner= True)
def load_data_for_visualization(nrows=None):
	data = pd.read_csv(FILE_NAME, parse_dates=True, index_col=DATE_COLUMN, nrows=nrows)
	data.rename(columns={"CH": "Curah Hujan"}, inplace=True)
	return data

def load_data_for_regression(nrows=None):
	data = pd.read_csv(FILE_NAME, nrows=nrows)
	# data.rename(columns={"CH": "Curah Hujan"}, inplace=True)
	return data

# PREPROCESSING DATA
def preprocessed_data(dataframe):
	# Fill missing values in QFF Column
	dataframe['QFF'].fillna(dataframe['QFF'].mean(), inplace=True)
	return dataframe

def split_dataset(dataset, test_ratio):
	test_indices = np.random.rand(len(dataset)) < test_ratio
	return dataset[~test_indices], dataset[test_indices]


def neural_network_model(features):
	normalizer = tf.keras.layers.Normalization(axis=-1)
	normalizer.adapt(np.array(features))
	model = tf.keras.Sequential([normalizer, 
									tf.keras.layers.Dense(128, activation=tf.nn.relu),
									tf.keras.layers.Dropout(0.5),
									tf.keras.layers.Dense(64, activation=tf.nn.relu),
									tf.keras.layers.Dropout(0.5),
									tf.keras.layers.Dense(1)])
	return model

def train_neural_network(dt_train_features, dt_train_labels, dt_test_features, dt_test_labels):
	early_stop = EarlyStopping(monitor='val_loss', mode="min", verbose=1, patience=10)
	model = neural_network_model(train_features)
	model.compile(metrics=["mae", "mse"],
				  loss=["mean_absolute_error", "mean_squared_error"],
				  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
	model.fit(train_features, train_labels, validation_split=0.1, epochs=500, verbose=1, callbacks=[early_stop])
	evaluation = model.evaluate(dt_test_features, dt_test_labels, verbose=0)
	return model, evaluation

def xgboost_model():
	hyper_params = {"tree_method":'gpu_hist',
					"n_estimators":2200, 
					"learning_rate":0.01,
					"seed":123,
					"max_depth":20,  
					"objective":'reg:squarederror',  
					"eval_metric":'mae',
					"verbosity":1,
					"booster":'gblinear',
					"reg_alpha":0.4640, 
					"reg_lambda":0.8571,
					"feature_selector":'cyclic',
					"random_state":111,
					"colsample_bytree":0.4603, 
					"gamma":0.0468, 
					"min_child_weight":1.7817,  
					"subsample":0.5213,
					"num_parallel_tree":1
	}
	model = XGBRegressor(**hyper_params)
	return model

def train_xgboost(df_train_features, df_test_features, df_train_label, df_test_label):
	model = xgboost_model()
	model.fit(df_train_features, df_train_label, verbose=2)
	predictions = model.predict(df_test_features)
	evaluation = [mean_absolute_error(df_test_label, predictions), mean_squared_error(df_test_label, predictions)]
	return model, evaluation

def lightgbm_model():
	hyper_params = {"task": "train",
					"boosting_type": "gbdt",
					"objective": "regression",
					"metric": ["l1", "l2"],
					"learning_rate": 0.001,
					"feature_fraction": 0.9,
					"bagging_fraction": 0.7,
					"bagging_freq": 10,
					"verbose": 0,
					"max_depth": 8,
					"num_leaves": 128,  
					"max_bin": 512,
					"num_iterations": 100000
					# "max_bin":55, 
					# "feature_fraction_seed":9, 
					# "bagging_seed":9,
					# "min_data_in_leaf":6, 
					# "min_sum_hessian_in_leaf":11,
	}
	model = lgb.LGBMRegressor(**hyper_params)
	return model

def train_lightgbm(df_train_features, df_test_features, df_train_label, df_test_label):
	model = lightgbm_model()
	model.fit(df_train_features, 
			  df_train_label,
			  eval_set=[(df_test_features, df_test_label)],
			  eval_metric='l2',
			  early_stopping_rounds=1000,
			  verbose=500
	)
	predictions = model.predict(df_test_features)
	evaluation = [mean_absolute_error(df_test_label, predictions), mean_squared_error(df_test_label, predictions)]
	return model, evaluation

def random_forest_model():
	model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
	return model

def train_random_forest(train_dataframe, test_dataframe, str_label):
	train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataframe=train_dataframe, 
													 label=str_label,
													 task=tfdf.keras.Task.REGRESSION)
	test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataframe=test_dataframe,
													label=str_label,
													task=tfdf.keras.Task.REGRESSION)
	model = random_forest_model()
	model.compile(metrics=["mae", "mse"])
	model.fit(x=train_ds)
	evaluation = model.evaluate(test_ds, return_dict=True)
	return model, evaluation
	
def gradient_boosting_model():
	model = tfdf.keras.GradientBoostingModel(task=tfdf.keras.Task.REGRESSION)
	return model

def train_gradient_boosting(train_dataframe, test_dataframe, str_label):
	train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataframe=train_dataframe, 
													 label=str_label,
													 task=tfdf.keras.Task.REGRESSION)
	test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataframe=test_dataframe,
													label=str_label,
													task=tfdf.keras.Task.REGRESSION)
	model = gradient_boosting_model()
	model.compile(metrics=["mae", "mse"])
	model.fit(x=train_ds)
	evaluation = model.evaluate(test_ds, return_dict=True)
	return model, evaluation

def input_prediction():
	with st.sidebar.expander("Nilai:", expanded=True):
		t0700 = st.number_input("T0700", value=30.1)
		t1300 = st.number_input("T1300", value=25.3)
		t1800 = st.number_input("T1800", value=26.8)
		trata = st.number_input("TRata-rata", value=17.4)
		tmax = st.number_input("Tmax", value=29.5)
		tmin = st.number_input("Tmin", value=22.4)
		lpm = st.number_input("LPM", value=48)
		qfe = st.number_input("QFE", value=1004.1)
		rh0700 = st.number_input("RH0700", value=93)
		rh1300 = st.number_input("RH1300", value=82)
		rh1800 = st.number_input("RH1800", value=77)
		rhrata = st.number_input("RHrata-rata", value=86)
		ffrata = st.number_input("ffrata-rata", value=4)
		dd = st.number_input("dd", value=270)
		ffmax = st.number_input("ffmax", value=10.0)
		ddmax = st.number_input("ddmax", value=270)
		qff = st.number_input("QFF", value=1010.36)

	# Store variables in record
	user_data = {
		"T0700": t0700,
		"T1300": t1300,
		"T1800": t1800,
		"TRata-rata": trata,
		"Tmax": tmax,
		"Tmin": tmin,
		"LPM": lpm,
		"QFE": qfe,
		"T1800": t1800,
		"RH0700": rh0700,
		"RH1300": rh1300,
		"RH1800": rh1800,
		"RHrata-rata": rhrata,	
		"ffrata-rata": ffrata,
		"dd": dd,
		"ffmax": ffmax,
		"ddmax": ddmax,
		"QFF": qff
	}

	input_features = pd.DataFrame(user_data, index=[0])
	return input_features

def save_evaluation_results(model_name, mae, mse, rmse, save_to_file=True):
	save_time = datetime.now()

	# If the save file exists, append the new results to the file
	# Store every evaluation result in dataframe, 
	# everytime the function is called, the dataframe will be appended
	# with the new evaluation result
	if save_to_file:
		try:
			if os.path.exists("db/evaluation_results.csv"):
				evaluation_results = pd.read_csv("db/evaluation_results.csv")
				evaluation_results = evaluation_results.append({"Date":save_time.strftime("%d/%m/%Y"),
																"Time":save_time.strftime("%H:%M:%S"),
																"Nama_Model": model_name, 
																"MAE": mae, 
																"MSE": mse, 
																"RMSE": rmse}, 
																ignore_index=True)
				evaluation_results.to_csv("db/evaluation_results.csv", index=False)
			else:
				evaluation_results = pd.DataFrame(columns=["Date", "Time", "model_name", "mae", "mse", "rmse"])
				evaluation_results = evaluation_results.append({"Date":save_time.strftime("%d/%m/%Y"),
																"Time":save_time.strftime("%H:%M:%S"),
																"Nama_Model": model_name, 
																"MAE": mae, 
																"MSE": mse, 
																"RMSE": rmse}, 
																ignore_index=True)
				evaluation_results.to_csv("db/evaluation_results.csv", index=False)
		# Raise error if failed to save the evaluation results
		except Exception as e:
			st.write("Gagal menyimpan file." + e)
			
	# return evaluation_results

def show_prediction(prediction):
	if prediction is None:
		st.warning("Belum melakukan prediksi ...")
	else:
		# st.caption("Hasil Prediksi")
		st.success(prediction)

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config( 
	page_title = "Peramalan Curah Hujan",
	initial_sidebar_state="expanded",
	layout="centered",
	page_icon="img/bmkg-icon.ico"
	)

st.image("img/banner.png", use_column_width="Auto")
st.title("Sistem Prakiraan Curah Hujan")	
st.write(
	"""
	##
	Laman ini menampilkan proses prakiraan curah hujan di Bandara Internasional I Gusti Ngurah Rai dan sekitarnya dengan bantuan model Machine Learning.
	"""
)

st.subheader("Pemilihan Parameter")
st.write(
	"""
	Bagian ini menampilkan parameter yang dipilih untuk membuat model peramalan.
	"""
	)
df = preprocessed_data(load_data_for_regression())
exclude_columns = ["Date", "Tahun", "Bulan", "Tanggal", "Cuaca Khusus", "CH"]
columns = [col for col in df.columns if col not in exclude_columns]
feature_select = st.multiselect(
	"Pilih:",
	columns
)
feature_select_all = st.checkbox("Pilih semua", value=False)
if feature_select_all:
	feature_select.clear()
	for col in columns:
		feature_select.append(col)

st.subheader("Proporsi Data Training")
data_training_slider = st.slider("Geser untuk menentukan nilai", 0, 100, 80)

# Create dataframe to store data training and testing proportion
data_training_df = pd.DataFrame({"Data Training (%)": [(data_training_slider)], 
									"Data Testing (%)": [100 - data_training_slider]})
st.dataframe(data_training_df)
data_test_ratio = (100 - data_training_slider) / 100

if not feature_select:
	st.warning("Tidak ada parameter yang dipilih")
	pass
else:
	used_feature = []

	for f in feature_select:
		# Feature column that used for predictions
		used_feature.append(f)
	
	# Add "Curah Hujan to used_feature object", because it's the target variable
	used_feature.append("CH")

	df_used = df.copy()
	df_used = df_used[used_feature]
	train_ds_pd, test_ds_pd = split_dataset(dataset=df_used, test_ratio=data_test_ratio)

	features = df_used.columns
	labels = ["CH"]
	features = [f for f in features if f not in labels]
	
	train_features = train_ds_pd[features].copy()
	test_features = test_ds_pd[features].copy()
	train_labels = train_ds_pd[labels].copy()
	test_labels = test_ds_pd[labels].copy()

# Create sidebar header
st.sidebar.subheader("Model Machine Learning")
training_result = pd.DataFrame(columns=["MAE", "MSE", "RMSE"])

model_ml = st.sidebar.radio(
	label="Pilih Model",
	options=("Tensorflow Deep Learning", "XGBoost Regressor", "LightGBM Regressor"),
	key="ml_radio_change"
)

# Create a button if the checkbox is checked
with st.sidebar.container():
	if st.sidebar.button(label="Training"):
		if model_ml == "Tensorflow Deep Learning":
			try:
				with st.spinner("Training Tensorflow Deep Learning"):
					tf.keras.backend.clear_session()
					nn_model, nn_evaluation = train_neural_network(train_features, train_labels, dt_test_features=test_features, dt_test_labels=test_labels)					
					st.success("Training Tensorflow Deep Learning Selesai")
					nn_model.save("models/neuralnetwork")
					nn_model.save_weights("models/neuralnetwork-weights")
					save_evaluation_results(model_ml, nn_evaluation[1], nn_evaluation[2],  np.sqrt(nn_evaluation[2]))
					# training_result.loc["Tensorflow Deep Learning"] = [nn_evaluation[1], nn_evaluation[2], np.sqrt(nn_evaluation[2])]
			except:
				st.warning("Belum memilih parameter ...")
		elif model_ml == "XGBoost Regressor":
			try:
				with st.spinner("Training XGBoost Regressor"):
					xgb_model, xgb_evaluation = train_xgboost(df_train_features=train_features, 
																df_test_features=test_features, 
																df_train_label=train_labels,  
																df_test_label=test_labels)
					st.success("Training XGBoost Regressor Selesai")
					xgb_model.save_model("models/xgboost.txt")
					# training_result.loc["XGBoost Regressor"] = [xgb_evaluation[0], xgb_evaluation[1], np.sqrt(xgb_evaluation[1])]
					save_evaluation_results(model_ml, xgb_evaluation[0], xgb_evaluation[1],  np.sqrt(xgb_evaluation[1]))
			except:
				st.warning("Belum memilih parameter ...")
		elif model_ml == "LightGBM Regressor":
			try:
				with st.spinner("Training LightGBM Regressor"):
					lgbm_model, lgbm_evaluation = train_lightgbm(df_train_features=train_features, 
																df_test_features=test_features, 
																df_train_label=train_labels,  
																df_test_label=test_labels)
					st.success("Training LightGBM Regressor Selesai")
					lgbm_model.save_model("models/lightgbm.txt")
					# print(lgbm_evaluation)
					# st.write(lgbm_evaluation)
					# training_result.loc["LightGBM Regressor"] = [lgbm_evaluation[0], lgbm_evaluation[1], np.sqrt(lgbm_evaluation[1])]
					save_evaluation_results(model_ml, lgbm_evaluation[0], lgbm_evaluation[1],  np.sqrt(lgbm_evaluation[1]))
			except:
				st.warning("Belum memilih parameter ...")

st.sidebar.subheader("Input Nilai Prakiraan")
df_input_prediction = input_prediction()

st.subheader("Contoh Data Prakiraan")
st.dataframe(df_input_prediction)

if st.sidebar.button(label="Prakirakan"):
	if model_ml == "Tensorflow Deep Learning":
		try:
			with st.spinner("Prakiraan Tensorflow Deep Learning"):
				# Load tensorflow model, then use it to predict the sidebar input data
				nn_model = tf.keras.models.load_model("models/neuralnetwork")
				nn_model.load_weights("models/neuralnetwork-weights")
				prediction = nn_model.predict(df_input_prediction).flatten()
				st.markdown("**Hasil Prakiraan Curah Hujan dengan Tensorflow Deep Learning**")
				show_prediction(np.round(prediction[0], 2))
		except: # 
			st.warning("Terjadi kesalahan proses prediksi...")
	elif model_ml == "XGBoost Regressor":
		try:
			with st.spinner("Prakiraan XGBoost Regressor"):
				xgb_model = XGBRegressor()
				xgb_model.load_model("models/xgboost.txt")
				prediction = xgb_model.predict(df_input_prediction)
				st.markdown("**Hasil Prakiraan Curah Hujan dengan XGBoost Regressor**")
				show_prediction(np.round(prediction[0], 2))
		except:
			st.warning("Terjadi kesalahan proses prediksi...")
	elif model_ml == "LightGBM Regressor":
		try:
			with st.spinner("Prakiraan LightGBM Regressor"):
				lgbm_model = lgb.Booster(model_file='models/lgbm-model-dummy.txt')
				prediction = lgbm_model.predict(df_input_prediction)
				st.markdown("**Hasil Prakiraan Curah Hujan dengan XGBoost Regressor**")
				show_prediction(np.round(prediction[0], 2))
		except:
			st.warning("Terjadi kesalahan proses prakiraan...")

st.subheader("Hasil Evaluasi Model")
try:	
	if os.path.exists("db/evaluation_results.csv"):
		evaluation = pd.read_csv("db/evaluation_results.csv")
		st.dataframe(evaluation)
except:
	st.warning("Belum melakukan training ...")