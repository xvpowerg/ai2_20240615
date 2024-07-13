import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title("圖表")

data = pd.DataFrame({
    "x":np.arange(1,101),
    "y":np.random.rand(100)
})

st.line_chart(data)
st.line_chart(data,x="x",y="y")
fig,ax = plt.subplots()
ax.plot(data["x"],data["y"],label="Test")
ax.legend()
st.pyplot(fig)