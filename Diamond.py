import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
pickle_in = open("Model/pipe1.pkl","rb")
regressor=pickle.load(pickle_in)

st.title("-----Case Study On Diamond Dataset----")
data=sns.load_dataset("diamonds")
st.write("shape of data",data.shape)
data["price"]=data["price"]*83.04

#--------------------------------------

menu=st.sidebar.radio("Menu",["Home","Prediction Price"])
if menu=="Home":
    st.image("diamond.jpeg")
    st.header("Basic Data Analysis  Operations On Tabular Data Of Diamond")
    if st.checkbox("Tabular Data of Diamond"):
        st.table(data.head(50))
    if st.checkbox("Statical summary of Diamond data"):
        st.table(data.describe())
    if st.checkbox("Correlation graph"):
       fig,ax=plt.subplots(figsize=(5,2.5))
       sns.heatmap(data.corr(),annot=True,cmap="coolwarm")
       st.pyplot(fig)

       #--------------------------------
    st.title("Univariate Graphs-")

    def categorical_data_count(y):
        fig,ax=plt.subplots(figsize=(5,2.5))
        st.write("countplot of",y," data")
        sns.countplot(x=data[y])
        st.pyplot(fig)
    graph=st.selectbox("categorical countplot",["Non","cut","color","clarity"])
    if graph=="Non":
        st.write("please select graphs in the the dropdown")
                                                 
    if graph=="cut":
        st.write(data["cut"].value_counts())
        categorical_data_count("cut")
    if graph=="color":
        st.write(data["color"].value_counts())
        categorical_data_count("color")
    if graph=="clarity":
        st.write(data["clarity"].value_counts())
        categorical_data_count("clarity")

    # st.title()
    def pie_chart(y):
        fig,ax=plt.subplots(figsize=(5,2.5))
        st.write("pie chart of",y," data")
        data[y].value_counts().plot(kind='pie',autopct='%.2f')
        st.pyplot(fig)
    graph=st.selectbox("pie chart-",["Non","cut","color","clarity"])
    if graph=="Non":
        st.write("please select graphs in the the dropdown")
    if graph=="cut":
        pie_chart("cut")
    if graph=="color":
        pie_chart("color")
    if graph=="clarity":
        pie_chart("clarity")

     # st.title()
    def histogram_plot(y):
        fig,ax=plt.subplots(figsize=(5,2.5))
        st.write("distplot of",y," data")
        sns.distplot(data[y],kde=True)
        st.pyplot(fig)
    graph=st.selectbox("Distplot-",["Non","carat","depth","table","price","x","y","z"])
    if graph=="Non":
        st.write("please select graphs in the the dropdown")
    if graph=="carat":
        histogram_plot("carat")
    if graph=="depth":
        histogram_plot("depth")
    if graph=="table":
        histogram_plot("table")
    if graph=="price":
        histogram_plot("price")
    if graph=="x":
        histogram_plot("x")
    if graph=="y":
        histogram_plot("y")
    if graph=="z":
        histogram_plot("z")

if menu=="Prediction Price":
    st.title("Diamond Price")
    html_temp = """
    <div style="background-color:orange;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Diamond Price Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    carat=st.slider("Enter carat",0.20,1.52)
    cut=st.radio("Pick your Diamond cut Quality",["Fair", "Good", "Very Good","Premium","Ideal"])
    color=st.radio("pick your color",['D','E','F','G','H','I','J'])
    clarity=st.radio("clarity of Diamond",["SI1","SI2","VS2","VS1","VVS2","VVS1","I1","IF"])
    depth=st.slider("Enter Depth",43.00,71.60)
    table=st.slider("Enter table",50.10,70.0)
    length=st.slider("Enter length",3.79,7.56)
    width=st.slider("Enter width",3.75,7.42)
    height=st.slider("Enter height",0.0,4.80)

    if st.button("predict"):
        test_input=np.array([carat,cut,color,clarity,depth,table,length,width,height],dtype=object).reshape(1,9)
        result=regressor.predict(test_input)
        st.success('The output is {}'.format(result))
  
