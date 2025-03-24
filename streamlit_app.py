# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
raw_data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Load model and artifacts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoders = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
with open('target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)

# Streamlit UI
st.title('Obesity Prediction App')
st.info('Predict your obesity level based on lifestyle and physical attributes.')

# Show Raw Data
with st.expander('Show Raw Data'):
    st.dataframe(raw_data, use_container_width=True)

# Data Visualization
with st.expander('Data Visualization'):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=raw_data, x='Height', y='Weight', hue='NObeyesdad', palette='Set2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    st.pyplot(fig)

# User Inputs
st.subheader('Input Your Data')
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 10, 100, 25)
height = st.number_input('Height (m)', min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, value=70.0)
family_history = st.selectbox('Family History with Overweight', ['yes', 'no'])
favc = st.selectbox('Frequent High Calorie Food Consumption', ['yes', 'no'])
fcvc = st.slider('Vegetable Consumption Frequency (1-3)', 1, 3, 2)
ncp = st.slider('Number of Main Meals', 1, 4, 3)
caec = st.selectbox('Food between Meals', ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox('Do you Smoke?', ['yes', 'no'])
ch2o = st.slider('Water Consumption (1-3)', 1, 3, 2)
scc = st.selectbox('Do you Monitor your Calories?', ['yes', 'no'])
faf = st.slider('Physical Activity Frequency (0-3)', 0, 3, 1)
tue = st.slider('Time Using Technology (0-3)', 0, 3, 1)
calc = st.selectbox('Alcohol Consumption', ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox('Transportation Mode', ['Automobile', 'Bike', 'Motorbike', 'Public Transportation', 'Walking'])

user_input = pd.DataFrame([{
    'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
    'family_history_with_overweight': family_history, 'FAVC': favc, 'FCVC': fcvc,
    'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o, 'SCC': scc,
    'FAF': faf, 'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
}])

st.subheader('Your Input Data')
st.dataframe(user_input, use_container_width=True)

def prepare_user_input(df):
    # Encode categorical features
    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    # Ensure feature order and fill missing
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    # Scale
    df_scaled = scaler.transform(df)
    return df_scaled

if st.button('Predict Obesity Level'):
    try:
        X_ready = prepare_user_input(user_input.copy())
        prediction = model.predict(X_ready)
        prediction_label = target_encoder.inverse_transform(prediction)
        proba = model.predict_proba(X_ready)

        st.success(f'Predicted Obesity Level: {prediction_label[0]}')
        st.info(f'Prediction Probability: {np.max(proba) * 100:.2f}%')

        # Show all class probabilities in a table
        st.subheader('Prediction Probabilities for All Classes')
        proba_df = pd.DataFrame({
            'Obesity Level': target_encoder.inverse_transform(np.arange(len(proba[0]))),
            'Probability (%)': (proba[0] * 100).round(2)
        }).sort_values(by='Probability (%)', ascending=False).reset_index(drop=True)
        st.dataframe(proba_df, use_container_width=True)

    except Exception as e:
        st.error(f'Prediction failed: {e}')
