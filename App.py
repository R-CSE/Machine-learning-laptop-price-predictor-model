import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load data and model
df = pd.read_csv("df.csv")
pipe = pickle.load(open("pipe.pkl", "rb"))

# Drop unnecessary columns
df = df[['Company', 'TypeName', 'Ram', 'Weight', 'Ips', 'ppi', 'Cpu_brand', 'HDD', 'SSD', 'Gpu_brand', 'os']]

st.title("💻 Laptop Price Predictor")

# Input features based on required columns
company = st.selectbox('Brand', df['Company'].unique())
lap_type = st.selectbox("Type", df['TypeName'].unique())
ram = st.selectbox("Ram (in GB)", sorted(df['Ram'].unique()))
weight = st.number_input("Weight of the Laptop (kg)", min_value=0.0, step=0.1)
ips_input = st.selectbox("IPS", ['No', 'Yes'])
screen_size = st.number_input('Screen Size (in inches)', min_value=1.0, step=0.1)
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
     '2880x1800', '2560x1600', '2560x1440', '2304x1440']
)
cpu = st.selectbox('CPU', df['Cpu_brand'].unique())
hdd = st.selectbox('HDD (in GB)', sorted(df['HDD'].unique()))
ssd = st.selectbox('SSD (in GB)', sorted(df['SSD'].unique()))
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())
os = st.selectbox('OS', df['os'].unique())

# Prediction
if st.button('Predict Price'):
    # Convert IPS input to binary
    ips = 1 if ips_input == 'Yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Prepare query in DataFrame format
    input_df = pd.DataFrame([[company, lap_type, ram, weight, ips, ppi,
                              cpu, hdd, ssd, gpu, os]],
        columns=['Company', 'TypeName', 'Ram', 'Weight', 'Ips', 'ppi',
                 'Cpu_brand', 'HDD', 'SSD', 'Gpu_brand', 'os']
    )

    # Predict and show result
    prediction = int(np.exp(pipe.predict(input_df)[0]))
    st.markdown(f"<h3 style='color:green; font-weight:bold;'>💰 The predicted price of this configuration is ₹{prediction}</h3>",
        unsafe_allow_html=True)

