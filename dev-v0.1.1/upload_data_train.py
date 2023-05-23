import pandas as pd
import data_processor as data_processor
import data_query as data_query
import streamlit as st

uploaded_file = st.file_uploader("Unggah file", type=["csv"])
if uploaded_file is not None:
	df = pd.read_csv(uploaded_file)
	df = df.rename(columns={'Trata-rata': 'Trata_rata', 
							'Cuaca Khusus': 'Cuaca_Khusus', 
							'RHrata-rata': 'RHrata_rata', 
							'ffrata-rata': 'ffrata_rata'
							})
	df_train, dt_test = data_processor.split_dataset(df)
	st.write(len(df_train))
	st.write(len(dt_test))

	if st.button("Simpan data"):
		data_query.insert_tb_train(df_train)
		data_query.insert_tb_test(dt_test)
		st.success("Data berhasil disimpan")
	