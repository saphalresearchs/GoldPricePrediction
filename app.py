import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')


random = joblib.load("random.pkl")
extra = joblib.load("extra.pkl")

st.title("Gold Price Prediction")
st.write("Write Features of Gold:")

spx = st.number_input("S&p 500 INDEX", format="%.4f")
uso = st.number_input("Oil Prices", format="%.4f")
sil = st.number_input("Silver", format="%.3f")
eur = st.number_input("Euro", format="%.5f")

if st.button("Predict"):
    input_data = [spx, uso, sil, eur]
    asarray = np.asarray(input_data)
    reshaped = asarray.reshape(1,-1)

    prediction1 = random.predict(reshaped)
    prediction2 = extra.predict(reshaped)

    avg = (prediction1[0]+prediction2[0])/2

    st.write("Price of gold is estimated to be:" , round(avg,2))
