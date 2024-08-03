import streamlit as st
import pandas as pd
import joblib
st.title("銷售預測應用")
model,scaler = joblib.load("scaled_sales_prediction_model.pkl")

# 輸入廣告支出
tv_spend = st.number_input('輸入 TV 廣告支出', min_value=0.0)
radio_spend = st.number_input('輸入 radio 廣告支出', min_value=0.0)
newspaper_spend = st.number_input('輸入 newspaper 廣告支出', min_value=0.0)

if st.button("預測銷售"):
    input_data ={
        "TV":tv_spend,
        "radio":radio_spend,
        "newspaper":newspaper_spend
    }
    input_df = pd.DataFrame([input_data])
    input_scaler = scaler.transform(input_df)
    ans = model.predict(input_scaler)[0]
    st.subheader("預測結果")
    st.write(f"預測銷售額是:{ans}")
    