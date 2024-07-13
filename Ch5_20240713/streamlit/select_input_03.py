import streamlit as st
from datetime import datetime
st.title("請選元素")
colorArray = ["紅","綠","藍"]
optionIndex =  st.selectbox("請選顏色",["紅","綠","藍"])
selectIndex = colorArray.index(optionIndex)
st.write("您選的是:",optionIndex,"SelectIndex:",selectIndex)


# 多選
fruits =  ['蘋果', '香蕉', '橙子', '葡萄', '西瓜']
optMult =  st.multiselect("喜歡的水果",fruits)
st.write("選的水果:",optMult)


#單選單
optionRadio =  st.radio("喜歡的季節", ['春季', '夏季', '秋季', '冬季'])
st.write("你選了:",optionRadio)

optionSilder = st.select_slider("選範圍",
                 options= [	'total_bill','tip','sex','smoker'	],value=('total_bill', 'smoker'))
st.write("選擇的範圍:",optionSilder)

valueSlider =  st.slider("數字範圍:",0.0,100.0,value=(25.0,75.0))
st.write("範圍:",valueSlider)
st.write("type:",type(valueSlider[0]))

checkBox =  st.checkbox("是否同意")
st.write("checkBox:",checkBox)
if checkBox:
    st.write("同意條款!")

dataInput = st.date_input("請選日期",datetime.today().date())
st.write("你選的日期:",dataInput)