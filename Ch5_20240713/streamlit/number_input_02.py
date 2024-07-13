import streamlit as st
st.title("Simple Number")

num1 = st.number_input("請輸入第一筆數字:",value=0,min_value=0,max_value=100)
num2 = st.number_input("請輸入第二筆數字:",value=5,min_value=2,max_value=200)

sum_result = num1 + num2

st.write(f"The sum is:{sum_result}")