import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Memuat dataset
dataset = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Memuat model dan alat bantu lainnya
with open('model.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    categorical_encoders = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    normalizer = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    selected_features = pickle.load(f)
with open('target_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# **Antarmuka Streamlit**
st.title('Aplikasi Prediksi Obesitas')
st.info('Prediksi tingkat obesitas berdasarkan gaya hidup dan karakteristik fisik.')

# **Menampilkan Data Mentah**
with st.expander('Lihat Data Mentah'):
    st.dataframe(dataset, use_container_width=True)

# **Visualisasi Data**
with st.expander('Visualisasi Data'):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=dataset, x='Height', y='Weight', hue='NObeyesdad', palette='Set2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

# **Input Data Pengguna**
st.subheader('Masukkan Data Anda')
gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
age = st.slider('Usia', 10, 100, 25)
height = st.number_input('Tinggi (m)', min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input('Berat (kg)', min_value=30.0, max_value=200.0, value=70.0)
family_history = st.selectbox('Riwayat Keluarga dengan Obesitas', ['yes', 'no'])
favc = st.selectbox('Sering Konsumsi Makanan Tinggi Kalori', ['yes', 'no'])
fcvc = st.slider('Frekuensi Konsumsi Sayuran (1-3)', 1, 3, 2)
ncp = st.slider('Jumlah Makan Utama per Hari', 1, 4, 3)
caec = st.selectbox('Konsumsi Makanan di Antara Waktu Makan', ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox('Apakah Anda Merokok?', ['yes', 'no'])
ch2o = st.slider('Konsumsi Air (1-3)', 1, 3, 2)
scc = st.selectbox('Apakah Anda Mengontrol Asupan Kalori?', ['yes', 'no'])
faf = st.slider('Frekuensi Aktivitas Fisik (0-3)', 0, 3, 1)
tue = st.slider('Waktu Penggunaan Teknologi (0-3)', 0, 3, 1)
calc = st.selectbox('Konsumsi Alkohol', ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox('Moda Transportasi', ['Automobile', 'Bike', 'Motorbike', 'Public Transportation', 'Walking'])

# Menyusun input pengguna dalam bentuk DataFrame
user_data = pd.DataFrame([{
    'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
    'family_history_with_overweight': family_history, 'FAVC': favc, 'FCVC': fcvc,
    'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o, 'SCC': scc,
    'FAF': faf, 'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
}])

st.subheader('Data yang Anda Masukkan')
st.dataframe(user_data, use_container_width=True)

# **Fungsi untuk Memproses Input Pengguna**
def process_input(df):
    # Mengonversi variabel kategori menggunakan encoder yang sudah dilatih
    for col, encoder in categorical_encoders.items():
        df[col] = encoder.transform(df[col])

    # Memastikan semua fitur tersedia dalam urutan yang benar
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0  # Mengisi fitur yang tidak ada dengan nilai nol
    df = df[selected_features]

    # Melakukan normalisasi
    df_normalized = normalizer.transform(df)
    return df_normalized

# **Prediksi Obesitas**
if st.button('Prediksi Tingkat Obesitas'):
    try:
        # Memproses data pengguna sebelum dimasukkan ke model
        prepared_input = process_input(user_data.copy())

        # Melakukan prediksi
        prediction = classifier.predict(prepared_input)
        predicted_label = label_encoder.inverse_transform(prediction)
        prediction_proba = classifier.predict_proba(prepared_input)

        # Menampilkan hasil prediksi
        st.success(f'Tingkat Obesitas yang Diprediksi: {predicted_label[0]}')
        st.info(f'Kepercayaan Model: {np.max(prediction_proba) * 100:.2f}%')

        # **Menampilkan Semua Probabilitas Kelas**
        st.subheader('Probabilitas untuk Setiap Kelas')
        probability_df = pd.DataFrame({
            'Tingkat Obesitas': label_encoder.inverse_transform(np.arange(len(prediction_proba[0]))),
            'Probabilitas (%)': (prediction_proba[0] * 100).round(2)
        }).sort_values(by='Probabilitas (%)', ascending=False).reset_index(drop=True)
        
        st.dataframe(probability_df, use_container_width=True)

    except Exception as e:
        st.error(f'Gagal melakukan prediksi: {e}')
