import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =============================
# Load trained model & columns
# =============================
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏠 HDB Resale Price Prediction")
st.write("Enter flat details to predict resale price.")

# =============================
# User Inputs
# =============================

month = st.text_input("Month (YYYY-MM)", "2023-01")
town = st.text_input("Town", "ANG MO KIO")
flat_type = st.selectbox("Flat Type", 
                         ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"])

flat_model = st.text_input("Flat Model", "Improved")

floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=30.0, max_value=200.0, value=90.0)

lease_commence_date = st.number_input("Lease Commence Year", min_value=1960, max_value=2025, value=2000)

storey_range = st.text_input("Storey Range (e.g. 01 TO 03)", "01 TO 03")

remaining_lease = st.text_input("Remaining Lease (e.g. 75 years)", "75 years")

# =============================
# Prediction
# =============================

if st.button("Predict Price"):

    # ===== Feature Engineering =====
    remaining_lease_years = int(''.join(filter(str.isdigit, remaining_lease)))
    avg_storey = int(storey_range.split(" ")[0])

    input_dict = {
        "month": month,
        "town": town,
        "flat_type": flat_type,
        "flat_model": flat_model,
        "floor_area_sqm": floor_area_sqm,
        "lease_commence_date": lease_commence_date,
        "avg_storey": avg_storey
    }

    df_input = pd.DataFrame([input_dict])

    # ===== Encode categorical features =====
    df_input = pd.get_dummies(df_input, drop_first=True)

    # ===== Match training columns =====
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    # ===== Predict =====
    prediction = model.predict(df_input)[0]

    st.success(f"Predicted Resale Price: ${prediction:,.2f}")
