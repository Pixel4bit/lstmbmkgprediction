# Parameter default
# link = https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv
# 

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import altair as alt
import time
import zipfile

# parameter

data_latih_x = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/data_latih_x.csv')
data_latih_y = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/data_latih_y.csv')
data_test_x = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/data_tes_x.csv')
data_test_y = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/data_tes_y.csv')
metrics_latih = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/metrics_latih.csv')
metrics_uji = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/metrics_uji.csv')



epoch = 75
batch = 32
val = 10

# Page title
st.set_page_config(page_title='BMKG LSTM Prediction', page_icon='📈')
st.title('📈 BMKG LSTM Prediction')

# Expander
with st.expander('**Tentang Website Ini**'):
  st.markdown('**Apa yang dilakukan website ini?**')
  st.info('Website ini hanya menampilkan hasil prediksi oleh model LSTM yang sudah dilatih sebelumnya.')

  st.markdown('**Bagaimana cara menggunakan Website ini?**')
  st.warning('Untuk menjalankan website ini cukup sederhana, pengguna hanya perlu mengatur parameter **jumlah hari** yang ingin diprediksi lalu scroll ke bawah lalu klik tombol **MULAI** untuk memulai proses inisialisasi model.  Sebagai hasilnya, website ini akan secara otomatis melakukan semua tahapan proses membangun model LSTM, dan menampilkan hasil prediksi model, evaluasi model, parameter model, dan dataset yang digunakan oleh model.')

  st.markdown('**Informasi tambahan**')
  st.markdown('Dataset:')
  st.code('''- Data Iklim harian BMKG Stasiun Meteorologi Kemayoran Jakarta Pusat
  ''', language='markdown')
  
  st.markdown('Library yang digunakan:')
  st.code('''- Pandas untuk analisa data dan manipulasi data
- Numpy untuk perhitungan statistik, matriks, dll.
- Keras untuk memuat model LSTM yang sudah dilatih
- Scikit-learn untuk proses normalisasi data dan evaluasi model LSTM
- Altair untuk grafik visualisasi
- Streamlit untuk user interface
  ''', language='markdown')



with st.expander('**Inisialisasi Model LSTM**'):
    st.info('Klik tombol MULAI dibawah ini untuk memulai proses inisialisasi model')
    example_data = st.button('MULAI')
    if example_data:
      climate_data = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/Raw_Dataset_BMKG_2013_2024_Jakarta_Pusat.csv')



# Sidebar for accepting input parameters
with st.sidebar:
    
    with st.expander('**Tentang Kami**'):
        st.info('Website ini dibuat oleh tiga mahasiswa dari Universitas Bina Sarana Informatika Prodi S1 Sistem Informasi')

    with st.expander('**Project**'):
        st.markdown('**Deep Learning**')
        st.info('Implementasi Model Deep Learning LSTM untuk memprediksi pola perubahan suhu di kota Jakarta Pusat')
        st.button('**Github**')

    st.header('Parameters')
    future = st.slider('Jumlah hari yang ingin diprediksi', 5, 390, 365, 5)

    sleep_time = st.slider('Sleep time', 0, 3, 0)
    
    parameter_split_size = 90
    parameter_n_estimators = 100
    parameter_max_features = 'all'
    parameter_min_samples_split = 2
    parameter_min_samples_leaf = 2

    parameter_random_state = 42
    parameter_criterion = 'squared_error'
    parameter_bootstrap = True
    parameter_oob_score = False

# Initiate the model building process
if example_data: 
    with st.status("Running ...", expanded=True) as status:
        
        # Reading data
        st.write("Loading data ...")
        time.sleep(sleep_time)

        # preprocessing data
        st.write("Preparing data ...")
        time.sleep(sleep_time + 1)

        # merubah format tanggal dataset agar sesuai
        climate_data['Tanggal'] = pd.to_datetime(climate_data['Tanggal'], format='%d/%m/%Y')

        # membuat kolom Tahun dengan mengambil Tahun dari kolom Tanggal
        climate_data['Tahun'] = climate_data['Tanggal'].dt.year

        # menghitung rata-rata tiap variabel per tahunnya, kecuali RR
        mean_pertahun = climate_data.groupby('Tahun').transform('mean')
        mean_pertahun.drop(columns=['RR'], inplace=True)

        # Mengisi semua missing values dengan rata-rata pertahun, kecuali RR
        climate_data = climate_data.fillna(mean_pertahun)

        # Mengisi missing values pada variabel RR dengan nilai 0 karena tidak setiap hari jakarta mengalami hujan
        modus = float(climate_data['RR'].mode())
        climate_data['RR'] = climate_data['RR'].fillna(modus)

        # Mengganti nilai 0 pada kolom 'Tn' dengan nilai rata-rata tahunan yang sesuai
        climate_data['Tn'] = climate_data.apply(lambda row: mean_pertahun.loc[row.name, 'Tn'] if row['Tn'] == 0 else row['Tn'], axis=1)
        
        # Mengubah kolom tanggal menjadi index karena ini merupakan data deret waktu
        climate_data.set_index('Tanggal', inplace=True)

        # menghapus kolom Tahun karena sudah tidak terpakai
        climate_data.drop(columns=['Tahun'], inplace=True)

        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LinearRegression

        x = climate_data.drop(columns=['Tx'], axis=1)
        y = climate_data['Tx']

        rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
        rfe.fit(x, y)
        
        #split data
        st.write("Splitting data ...")
        time.sleep(sleep_time)

        y_var = pd.DataFrame(y)
        kolom_terpilih = list(y_var.columns) + list(x.columns[rfe.ranking_ == 1])

        dataset = climate_data.astype(float) # membuat variabel baru untuk menyimpan dataset yang akan dilatih dan merubah nya data nya ke type float untuk kebutuhan proses kalkulasi agar lebih akurat
        dataset = dataset[kolom_terpilih] # Pemilihan kolom disesuaikan agar sama dengan kolom-kolom yang sudah terpilih dari metode seleksi RFE
        
        train_size = int(len(dataset) * 0.9)

        data_untuk_dilatih = dataset[:train_size]
        data_untuk_ditest = dataset[train_size:]
        #END OF CODE

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)

        st.write("Normalisasi data ...")
        time.sleep(sleep_time)

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_untuk_dilatih_scaled = scaler.fit_transform(data_untuk_dilatih)
        data_untuk_ditest_scaled = scaler.fit_transform(data_untuk_ditest)

        # Model training
        st.write("Model training ...")
        time.sleep(sleep_time + 1)

        # membuat set pelatihan
        # dengan contoh ini kita akan memprediksi data ke 15 dengan menggunakan data ke 0 - 14 untuk proses pelatihan.
        # kemudian mesin akan menggunakan data ke 1 - 15 untuk memprediksi data ke 16, begitu pula seterusnya.

        trainX = []
        trainY = []

        n_future = 1 # variabel yang akan memprediksi 1 hari kedepan untuk proses pelatihan
        n_past = 14 # variabel yang akan menggunakan 14 data terakhir untuk memprediksi data berikutnya,

        for i in range(n_past, len(data_untuk_dilatih_scaled) - n_future +1):
            trainX.append(data_untuk_dilatih_scaled[i - n_past:i, 0:data_untuk_dilatih.shape[1]])
            trainY.append(data_untuk_dilatih_scaled[i + n_future - 1:i + n_future, 0])

        trainX, trainY = np.array(trainX), np.array(trainY)


        custom_objects = {'mse': MeanSquaredError()}
        model = load_model('lstm.h5', custom_objects=custom_objects)
        model.summary()

        # END OF CODE
        if parameter_max_features == 'all':
            parameter_max_features = None
            parameter_max_features_metric = x.shape[1]
        
        rf = RandomForestRegressor(
                n_estimators=parameter_n_estimators,
                max_features=parameter_max_features,
                min_samples_split=parameter_min_samples_split,
                min_samples_leaf=parameter_min_samples_leaf,
                random_state=parameter_random_state,
                criterion=parameter_criterion,
                bootstrap=parameter_bootstrap,
                oob_score=parameter_oob_score)
        rf.fit(X_train, y_train)
        
        #prediksi
        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time + 2)

        # membuat set pengujian
        # dengan contoh ini kita akan memprediksi data ke 15 dengan menggunakan data ke 0 - 14 untuk proses pengujian.
        # kemudian mesin akan menggunakan data ke 1 - 15 untuk memprediksi data ke 16, begitu pula seterusnya.

        testX = []
        testY = []

        n_future = 1 # variabel yang akan memprediksi 1 hari kedepan untuk proses pengujian
        n_past = 14 # variabel yang akan menggunakan 14 data terakhir untuk memprediksi data berikutnya,

        for i in range(n_past, len(data_untuk_ditest_scaled) - n_future +1):
            testX.append(data_untuk_ditest_scaled[i - n_past:i, 0:data_untuk_ditest.shape[1]])
            testY.append(data_untuk_ditest_scaled[i + n_future - 1:i + n_future, 0])

        testX, testY = np.array(testX), np.array(testY)

        forecast_periode_tanggal = pd.date_range(list(climate_data.index)[-1], periods=future, freq='1d').tolist() # untuk mengambil periode 'tanggalan' dari dataset original yaitu climate_data

        forecast = model.predict(testX[-future:]) # melakukan proses prediksi 1 tahun ke masa depan

        # melakukan denormaliasi yaitu proses merubah nilai data ke bentuk/skala yang asli

        forecast_copies = np.repeat(forecast, dataset.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]

        # Membuat tabel untuk menyimpan data hasil prediksi agar lebih mudah untuk dilihat dan di plotting

        data_hasil = pd.DataFrame(y_pred_future, columns=['Prediksi']) # membuat konversi hasil dari perhitungan ke dalam bentuk tabel
        data_hasil['Tanggal'] = forecast_periode_tanggal # Menambahkan tanggalan agar data mudah dibaca
        data_hasil.set_index('Tanggal', inplace=True) # menjadikan tanggalan sebagai index karena dataset berupa timeseries

        mean_asli = climate_data['Tx'].mean() # mengambil nilai rata-rata dari kolom "Suhu tertinggi"
        mean_prediksi = data_hasil['Prediksi'].mean() # mengambil nilai rata-rata dari kolom "Prediksi"
        anomali_suhu_rata_rata = mean_prediksi - mean_asli

        selisih = pd.DataFrame(data=[[mean_prediksi, mean_asli, anomali_suhu_rata_rata]],
                       columns=['Tx_avg_prediksi', 'Tx_avg', 'Selisih / Anomali'])

        evaluasi_uji = model.predict(testX)
        evaluasi_uji_copies = np.repeat(evaluasi_uji, testX.shape[2], axis=-1)
        evaluasi_uji = scaler.inverse_transform(evaluasi_uji_copies)[:,0]

        testY = np.repeat(testY, testX.shape[2], axis=-1)
        testY = scaler.inverse_transform(testY)[:,0]

        dataY = pd.DataFrame(testY, columns=['Aktual'])
        dataY['Prediksi'] = evaluasi_uji

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mape = mean_absolute_percentage_error(testY, evaluasi_uji)
        mae = mean_absolute_error(testY, evaluasi_uji)
        mse = mean_squared_error(testY, evaluasi_uji)
        rmse = np.sqrt(mse)

        data = {
            "Metrics": ["MAE", "MAPE", "RMSE"],
            "Nilai": [mae, mape, rmse]
        }

        metrics_uji = pd.DataFrame(data)

        plot = alt.Chart(dataY.reset_index()).mark_line().encode(
            x=alt.X('index:N', title='Jumlah Data'),  # Mengubah judul sumbu x
            y=alt.Y('Aktual:Q', title='Suhu'),  # Mengubah judul sumbu y
            color=alt.value('blue')  # Warna biru untuk nilai aktual
        ).properties(
            title='Perbandingan Nilai Aktual dan Prediksi'
        ) + alt.Chart(dataY.reset_index()).mark_line().encode(
            x=alt.X('index:N', title='Jumlah Data'),  # Menggunakan judul sumbu x yang sama
            y=alt.Y('Prediksi:Q', title='Suhu'),  # Menggunakan judul sumbu y yang sama
            color=alt.value('red')  # Warna merah untuk nilai prediksi
        )

        # END OF CODE
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        
        # evaluasi
        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)

        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # evaluasi dipslay
        st.write("Displaying performance metrics ...")
        time.sleep(sleep_time)
        parameter_criterion_string = ' '.join([x.capitalize() for x in parameter_criterion.split('_')])
        #if 'Mse' in parameter_criterion_string:
        #    parameter_criterion_string = parameter_criterion_string.replace('Mse', 'MSE')
        rf_results = pd.DataFrame([train_mse, train_r2, test_mse, test_r2]).transpose()
        rf_results.columns = [f'Training {parameter_criterion_string}', 'Training R2', f'Test {parameter_criterion_string}', 'Test R2']
        # Convert objects to numerics
        for col in rf_results.columns:
            rf_results[col] = pd.to_numeric(rf_results[col], errors='ignore')
        # Round to 3 digits
        rf_results = rf_results.round(3)
        
    status.update(label="Status", state="complete", expanded=False)

    # Display data info
    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="Jumlah Sampel Data", value=x.shape[0], delta="")
    col[1].metric(label="Jumlah Variabel", value=climate_data.shape[1], delta="")
    col[2].metric(label="Jumlah Sampel Pelatihan", value=len(data_untuk_dilatih), delta="")
    col[3].metric(label="Jumlah Sampel Pengujian", value=len(data_untuk_ditest), delta="")
    
    with st.expander('Dataset Awal', expanded=True):
        st.dataframe(climate_data, height=210, use_container_width=True)
    with st.expander('Data latih', expanded=False):
        train_col = st.columns((3,1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(data_latih_x, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(data_latih_y, height=210, hide_index=True, use_container_width=True)
    with st.expander('Data uji', expanded=False):
        test_col = st.columns((3,1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(data_test_x, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(data_test_y, height=210, hide_index=True, use_container_width=True)

    # Download Zip dataset files
    climate_data.to_csv('dataset.csv', index=False)
    data_latih_x.to_csv('data_latih_x.csv', index=False)
    data_latih_y.to_csv('data_latih_y.csv', index=False)
    data_test_x.to_csv('data_test_x.csv', index=False)
    data_test_y.to_csv('data_test_y.csv', index=False)
    
    list_files = ['dataset.csv', 'data_latih_x.csv', 'data_latih_y.csv', 'data_test_x.csv', 'data_test_y.csv']
    with zipfile.ZipFile('dataset.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    with open('dataset.zip', 'rb') as datazip:
        btn = st.download_button(
                label='Download Data ZIP',
                data=datazip,
                file_name="dataset.zip",
                mime="application/octet-stream"
                )
    
    # Display model parameters
    st.header('Model parameters', divider='rainbow')
    parameters_col = st.columns(4)
    parameters_col[0].metric(label="Rasio Pelatihan (%)", value=90, delta="")
    parameters_col[1].metric(label="Jumlah Epoch", value=epoch, delta="")
    parameters_col[2].metric(label="Batch Size", value=batch, delta="")
    parameters_col[3].metric(label="Rasio Validasi (%)", value=val, delta="")
    
    # Display feature importance plot
    importances = rf.feature_importances_
    feature_names = list(x.columns)
    forest_importances = pd.Series(importances, index=feature_names)
    climate_data_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})
    
    bars = alt.Chart(climate_data_importance).mark_bar(size=40).encode(
             x='value:Q',
             y=alt.Y('feature:N', sort='-x')
           ).properties(height=250)

    st.header('Performa model', divider='rainbow')
    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.subheader('Pelatihan')
        st.dataframe(metrics_latih, use_container_width=True)
    with performance_col[2]:
        st.subheader('Pengujian')
        st.dataframe(metrics_uji, use_container_width=True)

    plt.figure(figsize=(10, 3))
    plt.plot(dataY['Aktual'], label='Aktual')
    plt.plot(dataY['Prediksi'], label='Prediksi')
    plt.title('Perbandingan Aktual vs Prediksi')
    plt.xlabel('Jumlah Data')
    plt.ylabel('Suhu')
    plt.legend()

    with st.expander('Akurasi Pengujian'):
            st.pyplot(plt, use_container_width=True)

    # Prediction results
    st.header('Hasil Prediksi', divider='rainbow')

    col = st.columns(4)
    col[0].metric(label="Jumlah Hari", value=future, delta="")
    col[1].metric(label="Suhu Terendah", value=round(float(data_hasil.min()), 2), delta="")
    col[2].metric(label="Suhu Tertinggi", value=round(float(data_hasil.max()), 2), delta="")
    col[3].metric(label="Suhu Rata-rata", value=round(float(data_hasil.mean()), 2), delta="")


    s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
    s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
    climate_data_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
    climate_data_train['class'] = 'train'
        
    s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
    s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
    climate_data_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
    climate_data_test['class'] = 'test'
    
    climate_data_prediction = pd.concat([climate_data_train, climate_data_test], axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(climate_data['Tx'][2950:], label='Data Historis')
    plt.plot(data_hasil['Prediksi'], label='Prediksi')
    plt.title(f'Hasil Prediksi Suhu Jakarta {future} hari')
    plt.xlabel('Tahun')
    plt.ylabel('Suhu')
    plt.legend()

    prediction_col = st.columns((2, 0.2, 3))
    
    # Display dataframe
    with prediction_col[0]:
        st.markdown('Data Prediksi')
        st.dataframe(data_hasil, height=320, use_container_width=True)

    # Display scatter plot of actual vs predicted values
    with prediction_col[2]:
        st.markdown('Visualisasi')
        st.pyplot(plt, use_container_width=True)

    
# Ask for CSV upload if none is detected
else:
    st.warning('👆🏻 Klik Inisialisasi Model LSTM diatas dan klik Tombol MULAI untuk memulai proses.')
