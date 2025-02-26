import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Real Estate System App - Gurgaon",
    page_icon="🏨",
)
#https://www.kaggle.com/code/datatech02/notebook18242af674/edit
st.markdown("# Real Estate System Project - VIT Research project")

PATH_Readme = Path("README.md")
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image('house.png')

with col3:
    st.write(' ')


try:
    st.markdown(PATH_Readme.read_text())
except FileNotFoundError:
    st.error("README.md file not found.", icon="🔥")