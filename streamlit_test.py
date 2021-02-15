import streamlit as st

import os
import time

import pandas as pd
import numpy as np
import plotly.express as px


#with st.echo():

tmp=pd.read_parquet("/Users/apple/Desktop/DataFiles/parts_forecasting/psa_orders/data.snappy.parquet")

st.dataframe(tmp.head())

st.markdown(" Test markdown ")


ff=px.bar(tmp.sample(2000),x='quarter',y='ORDER_QTY',color='ABC_Catg')

st.plotly_chart(ff)


st.markdown("option 2")
tmp_2=tmp.groupby(['quarter','FMS_Catg','ABC_Catg']).ORDER_QTY.sum().reset_index()
ff2=px.bar(tmp_2,x='quarter',y='ORDER_QTY',color='ABC_Catg')
st.plotly_chart(ff2)

with st.spinner('Wait for it...'):
    time.sleep(3)
st.success('Done!')
