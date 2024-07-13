import streamlit as st
import pandas as pd

st.title("讀取CSV")

updateFile = st.file_uploader("選擇一個CSV",type="csv")

if updateFile is not None:
    df = pd.read_csv(updateFile)
    st.subheader("原始數據")
    st.dataframe(df)
    
    st.subheader("數據的資訊")
    st.dataframe(df.describe())

    st.subheader("繪製圖片")
    st.line_chart(df)

    colum = st.selectbox("請選欄位",df.columns)
    st.dataframe(df[[colum]])