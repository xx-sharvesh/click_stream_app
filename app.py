import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pipeline_feature_engineering import transform_input  # our pipeline function

st.set_page_config(page_title="Clickstream Conversion App", layout="wide")
st.title("🛒 Customer Conversion & Revenue App")

# Load trained models
clf = joblib.load("rf_classifier.pkl")
regr = joblib.load("rf_regressor.pkl")
scaler, kmeans = joblib.load("kmeans_model.pkl")

st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose Input Mode:", ["Upload CSV", "Manual Entry"])

if input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload raw clickstream CSV", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview", raw_df.head())
        features_df = transform_input(raw_df)
        st.write("### Engineered Features", features_df.head())

        # Drop session_id for prediction
        X = features_df.drop(columns=['session_id','purchase_flag','total_spend'])
        # Predictions
        st.subheader("Predictions")
        pred_class = clf.predict(X)
        pred_reg = regr.predict(X)
        pred_cluster = kmeans.predict(scaler.transform(X))

        result_df = features_df.copy()
        result_df['Predicted_Conversion'] = pred_class
        result_df['Predicted_Revenue'] = pred_reg
        result_df['Cluster_Label'] = pred_cluster
        st.dataframe(result_df)

elif input_mode == "Manual Entry":
    st.write("Enter single record of raw clickstream data:")
    # minimal fields for one click row:
    year = st.number_input("Year", min_value=2000, max_value=2100, value=2008)
    month = st.number_input("Month", min_value=1, max_value=12, value=4)
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
    order = st.number_input("Order (click seq)", min_value=0, value=1)
    country = st.number_input("Country code", min_value=1, max_value=47, value=1)
    session_id = st.text_input("Session ID", value="1")
    page1_main_category = st.number_input("Main category", min_value=1, max_value=4, value=1)
    page2_clothing_model = st.text_input("Clothing model code", value="A1")
    colour = st.number_input("Colour code", min_value=1, max_value=14, value=1)
    location = st.number_input("Location", min_value=1, max_value=6, value=1)
    model_photography = st.number_input("Model photo type", min_value=1, max_value=2, value=1)
    price = st.number_input("Price", min_value=0.0, value=10.0)
    price_2 = st.number_input("Price above avg? 1=yes 2=no", min_value=1, max_value=2, value=1)
    page = st.number_input("Page number", min_value=1, max_value=5, value=1)

    if st.button("Predict"):
        raw_df = pd.DataFrame([{
            'year':year,'month':month,'day':day,'order':order,'country':country,
            'session_id':session_id,'page1_main_category':page1_main_category,
            'page2_clothing_model':page2_clothing_model,'colour':colour,
            'location':location,'model_photography':model_photography,'price':price,
            'price_2':price_2,'page':page
        }])

        features_df = transform_input(raw_df)

        X = features_df.drop(columns=['session_id','purchase_flag','total_spend'])
        pred_class = clf.predict(X)[0]
        pred_reg = regr.predict(X)[0]
        pred_cluster = kmeans.predict(scaler.transform(X))[0]

        st.success(f"Predicted Conversion: {pred_class}")
        st.info(f"Predicted Revenue: {pred_reg:.2f}")
        st.warning(f"Cluster Label: {pred_cluster}")
