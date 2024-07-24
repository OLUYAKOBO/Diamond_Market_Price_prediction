import pandas as pd
import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

scaler = pickle.load(open('scal.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))
#model = pickle.load(open('diamond.pkl','rb'))
st.title(" *Diamond Price Prediction Application*")
st.write("*This app predicts the market value of **Diamonds***")

st.header("*Enter the details of the Diamond below*")

def user_input():
    c1,c2 =  st.columns(2)
    with c1:
        
        carat = st.number_input('*How many carat is the Diamond*',0.2,3.65,1.3)
        cut  = st.selectbox('*What type of cut is the Diamond*',(['Ideal','Premium','Very Good','Good','Fair']))
        color = st.selectbox('*What is the color type of the Diamond*',(['D','E','F','G','H','I','J']))
        
    with c2:
        
        clarity = st.selectbox('*Select the clarity of the Diamond*', (['SI1', 'VS2','SI2','VS1','VVS2','VVS1','IF','I1']))
        depth = st.number_input("*What is the depth of the Diamond*", 43.0,79.0, 46.7)
        table = st.number_input("*What is the table value of the Diamond*", 43.0,79.0, 46.7)
        
    feat = np.array([carat,cut,color,clarity,depth,table]).reshape(1,-1)
    cols = ['carat','cut','color','clarity','depth','table']
    feat1 = pd.DataFrame(feat, columns=cols)
    return feat1

df = user_input()
#st.write(df)

num_features = df.select_dtypes(exclude = 'object').columns
cat_features = df.select_dtypes(include = 'object').columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler

def prepare(df):
    cat_cols = ['cut','color','clarity']
    df1 = df.copy()
    encoded_data = encoder.transform(df1[cat_cols])
    dense_data = encoded_data.todense()
    df1_encoded = pd.DataFrame(dense_data, columns = encoder.get_feature_names_out())
    
    df1 = pd.concat([df1,df1_encoded],
                    axis = 1)
    df1.drop(cat_cols,
             axis = 1,
             inplace = True)
    
    
    cols = df1.columns
    df1 = scaler.transform(df1)
    df1 = pd.DataFrame(df1,columns=cols)
    return df1
df1 = prepare(df)

#st.write(df1)                                                                      
model = pickle.load(open('cat_diamond.pkl','rb'))
predictions = model.predict(df1)

#st.subheader('*Diamond Price*')
#if st.button('*Click here to get the price of the **Diamond***'):
    #st.write(predictions)
    


import time

if st.button('*Click here to get the price of the **Diamond***'):
    time.sleep(10)
    with st.spinner('Predicting... Please wait...'):
        st.success(f'Your Diamond is valued at {predictions.item()}')
        
