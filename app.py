import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from geopy.distance import geodesic

# ✅ Load model
model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoder.jb")

# ✅ Function
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1),(lat2,lon2)).km

# ✅ UI
st.title("Fraud Detection System")
st.write("App running")
st.write("Enter the Transaction details Below")

# ✅ Inputs
merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
lat = st.number_input("Latitude",format="%.6f")
long = st.number_input("Longitude",format="%.6f")
merch_lat = st.number_input("Merchant Latitude",format="%.6f")
merch_long = st.number_input("Merchant Longitude",format="%.6f")
hour = st.slider("Transaction Hour",0,23,12)
day = st.slider("Transaction Day",1,31,15)
month = st.slider("Transaction Month",1,12,6)
gender = st.selectbox("Gender",["Male","Female"])
cc_num = st.text_input("Credit Card number")

# ✅ Distance
distance = haversine(lat,long,merch_lat,merch_long)

# 🚀 Button
if st.button("Check For Fraud"):
    if merchant and category and cc_num:

        st.write(f"📍 Distance: {distance:.2f} km")

        # 🚨 Fraud reasons
        if distance > 1000:
            st.warning("⚠️ Large distance detected")

        if amt > 10000:
            st.warning("⚠️ High transaction amount")

        # 🌍 Google Maps links
        user_map_url = f"https://www.google.com/maps?q={lat},{long}"
        merchant_map_url = f"https://www.google.com/maps?q={merch_lat},{merch_long}"

        st.write("📍 View Locations on Google Maps:")
        st.markdown(f"🔵 [Open User Location]({user_map_url})")
        st.markdown(f"🔴 [Open Merchant Location]({merchant_map_url})")

        # 📊 Prepare data
        input_data = pd.DataFrame(
            [[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
            columns=['merchant','category','amt','distance','hour','day','month','gender','cc_num']
        )

        # 🔁 Encoding
        categorical_col = ['merchant','category','gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        # 🔢 Hash card
        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        # 🤖 Prediction
        prediction = model.predict(input_data)[0]

        # 📊 Probability
        prob = model.predict_proba(input_data)[0][1]
        st.write(f"📊 Fraud Probability: {prob*100:.2f}%")

        # 🎯 Risk Level
        if prob > 0.8:
            st.error("🔴 High Risk Transaction")
        elif prob > 0.5:
            st.warning("🟡 Medium Risk Transaction")
        else:
            st.success("🟢 Low Risk Transaction")

        # 🚨 Final decision (ML + Rules)
        if prediction == 1 or (distance > 1000 and amt > 10000):
            st.error("⚠️ Fraud Detected! This transaction is suspicious.")
        else:
            st.success("✅ Safe Transaction")

    else:
        st.error("Please Fill all required fields")

