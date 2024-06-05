# Parameter default
# link = https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv
# 

import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
plt.style.use('seaborn-v0_8-notebook')

from keras.models import load_model
from keras.losses import MeanSquaredError

import time
import zipfile

# parameter

data_latih_x = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/data_latih_x.csv')
data_latih_y = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/data_latih_y.csv')
data_test_x = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/data_tes_x.csv')
data_test_y = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/data_tes_y.csv')
metrics_latih = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/metrics_latih.csv')
metrics_uji = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/metrics_uji.csv')
dataX = pd.read_csv('https://raw.githubusercontent.com/Pixel4bit/Data-BMKG/main/hasil/csv/dataX.csv')


size_latih = 90
epoch = 75
batch = 32
val = 10

# Page title
st.set_page_config(page_title='BMKG LSTM Prediction', page_icon='📈')
st.title('📈 BMKG Prediction with Deep Learning algorithm')

# Expander
with st.expander('🌐 **Tentang Website Ini**'):
  st.markdown('**Apa yang dilakukan website ini?**')
  st.info('Website ini akan menampilkan hasil prediksi oleh model deep learning dengan algoritma LSTM yang sudah dilatih sebelumnya.')

  st.markdown('**Bagaimana cara menggunakan Website ini?**')
  st.warning('Untuk menjalankan website ini cukup sederhana, pengguna hanya perlu mengatur parameter **jumlah hari** yang ingin diprediksi lalu scroll ke bawah kemudian klik tombol **MULAI** untuk memulai proses inisialisasi model.  Sebagai hasilnya, website ini akan secara otomatis melakukan semua tahapan proses membangun model LSTM, dan menampilkan hasil prediksi model, evaluasi model, parameter model, dan dataset yang digunakan oleh model.')

  st.markdown('**Informasi tambahan**')
  st.markdown('Dataset:')
  st.code('''- Data Iklim harian BMKG Stasiun Meteorologi Kemayoran Jakarta Pusat
  - Rentang Waktu dataset: 01 Januari 2013 s.d 01 Februari 2024
  ''', language='markdown')
  
  st.markdown('Library yang digunakan:')
  st.code('''- PANDAS untuk analisa data dan manipulasi data
- NUMPY untuk perhitungan statistik, matriks, dll.
- KERAS untuk memuat model LSTM yang sudah dilatih
- SCIKIT-LEARN untuk proses normalisasi data dan evaluasi model LSTM
- ALTAIR untuk grafik visualisasi
- STREAMLIT untuk user interface
  ''', language='markdown')



with st.expander('🤖 **Inisialisasi Model LSTM**', expanded=True):
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
    future = st.slider('Jumlah hari yang ingin diprediksi ke masa depan', 5, 365, 365, 5)
    
    n_past = st.slider('Pola data masa lalu yang akan dipelajari oleh model LSTM', 5, 40, 14, 1)

    sleep_time = st.slider('Sleep time', 0, 3, 0)

# Initiate the model building process
if example_data: 
    with st.status("Running ...", expanded=True) as status:
        
        # Reading data
        st.write("Loading data ...")
        waktu_mulai = time.time()
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

        y = pd.DataFrame(y)
        kolom_terpilih = list(y.columns) + list(x.columns[rfe.ranking_ == 1])

        dataset = climate_data.astype(float) # membuat variabel baru untuk menyimpan dataset yang akan dilatih dan merubah nya data nya ke type float untuk kebutuhan proses kalkulasi agar lebih akurat
        dataset = dataset[kolom_terpilih] # Pemilihan kolom disesuaikan agar sama dengan kolom-kolom yang sudah terpilih dari metode seleksi RFE
        
        train_size = int(len(dataset) * 0.9)

        data_untuk_dilatih = dataset[:train_size]
        data_untuk_ditest = dataset[train_size:]

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
        #n_past = 14 # variabel yang akan menggunakan 14 data terakhir untuk memprediksi data berikutnya,

        for i in range(n_past, len(data_untuk_dilatih_scaled) - n_future +1):
            trainX.append(data_untuk_dilatih_scaled[i - n_past:i, 0:data_untuk_dilatih.shape[1]])
            trainY.append(data_untuk_dilatih_scaled[i + n_future - 1:i + n_future, 0])

        trainX, trainY = np.array(trainX), np.array(trainY)


        custom_objects = {'mse': MeanSquaredError()}
        model = load_model('lstm.h5', custom_objects=custom_objects)
        
        #prediksi
        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time + 2)

        # membuat set pengujian
        # dengan contoh ini kita akan memprediksi data ke 15 dengan menggunakan data ke 0 - 14 untuk proses pengujian.
        # kemudian mesin akan menggunakan data ke 1 - 15 untuk memprediksi data ke 16, begitu pula seterusnya.

        testX = []
        testY = []

        n_future = 1 # variabel yang akan memprediksi 1 hari kedepan untuk proses pengujian
        #n_past = 14 # variabel yang akan menggunakan 14 data terakhir untuk memprediksi data berikutnya,

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
        
        # evaluasi
        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        
        evaluasi_latih = model.predict(trainX)
        evaluasi_latih_copies = np.repeat(evaluasi_latih, trainX.shape[2], axis=-1)
        evaluasi_latih = scaler.inverse_transform(evaluasi_latih_copies)[:,0]

        trainY = np.repeat(trainY, trainX.shape[2], axis=-1)
        trainY = scaler.inverse_transform(trainY)[:,0]

        dataX = pd.DataFrame(trainY, columns=['Aktual'])
        dataX['Prediksi'] = evaluasi_latih
        
        evaluasi_uji = model.predict(testX)
        evaluasi_uji_copies = np.repeat(evaluasi_uji, testX.shape[2], axis=-1)
        evaluasi_uji = scaler.inverse_transform(evaluasi_uji_copies)[:,0]

        testY = np.repeat(testY, testX.shape[2], axis=-1)
        testY = scaler.inverse_transform(testY)[:,0]

        dataY = pd.DataFrame(testY, columns=['Aktual'])
        dataY['Prediksi'] = evaluasi_uji
        
        # evaluasi dipslay
        st.write("Displaying performance metrics ...")
        time.sleep(sleep_time)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mape = mean_absolute_percentage_error(trainY, evaluasi_latih)
        mae = mean_absolute_error(trainY, evaluasi_latih)
        mse = mean_squared_error(trainY, evaluasi_latih)
        rmse = np.sqrt(mse)

        data = {
            "Metrics": ["MAE", "MAPE", "RMSE"],
            "Nilai": [mae, mape, rmse]
        }

        metrics_latih = pd.DataFrame(data)

        mape = mean_absolute_percentage_error(testY, evaluasi_uji)
        mae = mean_absolute_error(testY, evaluasi_uji)
        mse = mean_squared_error(testY, evaluasi_uji)
        rmse = np.sqrt(mse)

        data = {
            "Metrics": ["MAE", "MAPE", "RMSE"],
            "Nilai": [mae, mape, rmse]
        }

        metrics_uji = pd.DataFrame(data)

        waktu_berakhir = time.time()
        durasi = float(waktu_berakhir - waktu_mulai)
        
    status.update(label="Status", state="complete", expanded=False)
    st.write(f'*Running time: {round(durasi, 2)} detik*')

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
    
    plt.figure(figsize= (20,7), dpi=200)
    plt.title('Perbandingan Data latih dan Data Uji', fontsize=20)
    plt.plot(climate_data['Tx'][:train_size], label='Data Latih')
    plt.plot(climate_data['Tx'][train_size:], label='Data Uji')
    plt.xlabel('Tahun', fontsize=14)
    plt.ylabel('Suhu', fontsize=14)
    plt.legend(loc='lower right', fontsize=14)
    
    with st.expander('Perbandingan Data latih dan Uji'):
        st.pyplot(plt, use_container_width=True)
        

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
    parameters_col[0].metric(label="Rasio Pelatihan", value=f'{size_latih}%', delta="")
    parameters_col[1].metric(label="Jumlah Epoch", value=epoch, delta="")
    parameters_col[2].metric(label="Batch Size", value=batch, delta="")
    parameters_col[3].metric(label="Rasio Validasi", value=f'{val}%', delta="")

    st.header('Performa model', divider='rainbow')
    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.subheader('Pelatihan')
        st.info('Rendah lebih baik')
        st.dataframe(metrics_latih, use_container_width=True)
    with performance_col[2]:
        st.subheader('Pengujian')
        st.info('Rendah lebih baik')
        st.dataframe(metrics_uji, use_container_width=True)
        
    plt.rcParams['font.size'] = 5
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = (10, 3)

    plt.figure(figsize=(10, 3), dpi=200)
    plt.plot(dataX['Aktual'][::10], label='Aktual')
    plt.plot(dataX['Prediksi'][::10], label='Prediksi')
    plt.title('Perbandingan Data Latih Aktual vs Prediksi')
    plt.xlabel('Jumlah Data')
    plt.ylabel('Suhu')
    plt.legend(loc='lower right')

    with st.expander('Akurasi Pelatihan'):
            st.pyplot(plt, use_container_width=True)

    plt.figure(figsize=(10, 3))
    plt.plot(dataY['Aktual'], label='Aktual')
    plt.plot(dataY['Prediksi'], label='Prediksi')
    plt.title('Perbandingan Data Uji Aktual vs Prediksi')
    plt.xlabel('Jumlah Data')
    plt.ylabel('Suhu')
    plt.legend(loc= 'upper left')

    with st.expander('Akurasi Pengujian'):
            st.pyplot(plt, use_container_width=True)

    # Prediction results
    st.header('Hasil Prediksi', divider='rainbow')

    col = st.columns(4)
    col[0].metric(label="Jumlah Hari", value=future, delta="")
    col[1].metric(label="Suhu Terendah", value=f'{round(float(data_hasil.min()), 2)}°', delta="")
    col[2].metric(label="Suhu Tertinggi", value=f'{round(float(data_hasil.max()), 2)}°', delta="")
    col[3].metric(label="Suhu Rata-rata", value=f'{round(float(data_hasil.mean()), 2)}°', delta="")
    
    plt.figure(figsize=(10, 4), dpi=200)
    plt.plot(climate_data['Tx'][2950:], label='Data Historis')
    plt.plot(data_hasil['Prediksi'], label='Prediksi')
    plt.gca().set_facecolor(color='white')
    plt.gcf().set_facecolor(color='white')
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
