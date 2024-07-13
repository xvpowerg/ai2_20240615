import streamlit as st
st.title("Hello Streamlit")

user_input = st.text_input("Type Something....")

st.write(f"你的輸入{user_input}")
