 -- Create table tb_train contain
 -- Date in Date format, Tahun integer, Bulan integer, Tanggal integer,
 -- T0700 float, T1300 float, T1800 float, Trata_rata float, Tmax float, Tmin float,
 -- CH float, LPM float, Cuaca_Khusus varchar, QFE float, RH0700 float, RH1300 float,
 -- RH1800 float, RHrata_rata float, ffrata_rata float, dd float, ffmax float,
 -- ddmax float, QFF float

 CREATE TABLE tb_train (
 	Date date,
	Tahun int,
	Bulan int,
	Tanggal int, 
 	T0700 float,
 	T1300 float,
 	T1800 float,
 	Trata_rata float,
 	Tmax float,
 	Tmin float,
	CH float,
	LPM float,
	Cuaca_Khusus varchar,
	QFE float,
	RH0700 float,
	RH1300 float,
	RH1800 float,
	RHrata_rata float,
	ffrata_rata float,
	dd float,
	ffmax float,
	ddmax float,
	QFF float
	);

 -- Create table tb_test contain 
 -- Date in Date format, Tahun integer, Bulan integer, Tanggal integer,
 -- T0700 float, T1300 float, T1800 float, Trata_rata float, Tmax float, Tmin float,
 -- CH float, LPM float, Cuaca_Khusus varchar, QFE float, RH0700 float, RH1300 float,
 -- RH1800 float, RHrata_rata float, ffrata_rata float, dd float, ffmax float,
 -- ddmax float, QFF float
  CREATE TABLE tb_test (
 	Date date,
	Tahun int,
	Bulan int,
	Tanggal int, 
 	T0700 float,
 	T1300 float,
 	T1800 float,
 	Trata_rata float,
 	Tmax float,
 	Tmin float,
	CH float,
	LPM float,
	Cuaca_Khusus varchar,
	QFE float,
	RH0700 float,
	RH1300 float,
	RH1800 float,
	RHrata_rata float,
	ffrata_rata float,
	dd float,
	ffmax float,
	ddmax float,
	QFF float
	);

-- Create table tb_train_result contain
CREATE TABLE tb_train_result (
	date date,
	time varchar,
	model text,
	jml_atribut int,
	atribut varchar,
	mae float,
	mse float,
	rmse float
	);

-- Create table tb_train_result contain
CREATE TABLE tb_test_result (
	date date,
	time varchar,
	model text,
	jml_atribut int,
	atribut varchar,
	mae float,
	mse float,
	rmse float
	);

