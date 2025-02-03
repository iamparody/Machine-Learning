import streamlit as st
import pickle
import numpy as np
import json

# Load model and data columns
with open("model/bhk.pickle", "rb") as f:
    model = pickle.load(f)

with open("model/columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]
    locations = data_columns[3:]

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", page_icon="\U0001F3E0", layout="centered")
st.title("🏡 House Price Prediction App")
st.write("Fill in the details below to get an estimated house price!")

# Inputs
location = st.selectbox("Select Location", locations)
total_sqft = st.slider("Total Square Feet", min_value=300, max_value=5000, step=50, value=1000)
bhk = st.select_slider("BHK (Bedrooms)", options=[1, 2, 3, 4, 5], value=3)
bath = st.select_slider("Bathrooms", options=[1, 2, 3, 4, 5], value=2)

# Predict Function
def predict_price(location, sqft, bhk, bath):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bhk
    x[2] = bath
    if location in locations:
        loc_index = data_columns.index(location)
        x[loc_index] = 1
    return round(model.predict([x])[0], 2)

# Prediction Button
if st.button("Predict Price 💰"):
    price = predict_price(location, total_sqft, bhk, bath)
    st.success(f"Estimated Price: **KSh {price} Lakhs**")
