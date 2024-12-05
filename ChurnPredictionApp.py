import streamlit as st
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd 
import pickle as pkl
from tensorflow.keras.models import load_model

if 'columns' not in st.session_state:
    df = pd.read_csv(r"Churn_Modelling.csv")
    df = df.iloc[:,3:-1]
    df = pd.get_dummies(df)
    st.session_state['columns'] = df.columns.tolist()

if 'geography' not in st.session_state:
    df = pd.read_csv(r"Churn_Modelling.csv")
    st.session_state['geography'] = df['Geography'].unique()

if 'gender' not in st.session_state:
    df = pd.read_csv(r"Churn_Modelling.csv")
    st.session_state['gender'] = df['Gender'].unique()

if 'age' not in st.session_state:
    df = pd.read_csv(r"Churn_Modelling.csv")
    st.session_state['age'] = (df['Age'].min(), df['Age'].max(), df['Age'].mean())

if 'balance' not in st.session_state:
    df = pd.read_csv(r"Churn_Modelling.csv")
    st.session_state['balance'] = (df['Balance'].min(), df['Balance'].max(), df['Balance'].mean())

if 'credit_score' not in st.session_state:
    df = pd.read_csv(r"Churn_Modelling.csv")
    st.session_state['credit_score'] = (df['CreditScore'].min(), df['CreditScore'].max(), df['CreditScore'].mean())

if 'tenure' not in st.session_state:
    df = pd.read_csv(r"Churn_Modelling.csv")
    st.session_state['tenure'] = (df['Tenure'].min(), df['Tenure'].max(), df['Tenure'].mean())

if 'NumOfProducts' not in st.session_state:
    df = pd.read_csv(r"Churn_Modelling.csv")
    st.session_state['NumOfProducts'] = (df['NumOfProducts'].min(), df['NumOfProducts'].max(), df['NumOfProducts'].mean())

if 'HasCrCard' not in st.session_state:
    st.session_state['HasCrCard'] = (0,1)

if 'IsActiveMember' not in st.session_state:
    st.session_state['IsActiveMember'] = (0,1)

if 'EstimatedSalary' not in st.session_state:
    df = pd.read_csv(r"Churn_Modelling.csv")
    st.session_state['EstimatedSalary'] = (df['EstimatedSalary'].min(), df['EstimatedSalary'].max(), df['EstimatedSalary'].mean())

# load the trained model 
model = tf.keras.models.load_model(r"churn_model.h5")

# load the scalers and encoders
with open(r"labelEncoder.pkl",'rb') as f:
    labelEncoder = pkl.load(f)


with open(r"standardSacler.pkl",'rb') as f:
    standardSacler = pkl.load(f)

st.title('Customer Churn Prediction')

with st.sidebar:
    vGeography = st.selectbox('Geography',st.session_state['geography'])
    vGender = st.selectbox('Gender',st.session_state['gender'])
    vAge = st.slider('Age',st.session_state['age'][0],st.session_state['age'][1],value=int(st.session_state['age'][2]))
    vBalance = st.slider('Balance',int(st.session_state['balance'][0]),int(st.session_state['balance'][1]),value=int(st.session_state['balance'][2]))
    vCreditScore = st.slider('Credit Score',st.session_state['credit_score'][0],st.session_state['credit_score'][1],value=int(st.session_state['credit_score'][2]))
    vTenure = st.slider('tenure',int(st.session_state['tenure'][0]),int(st.session_state['tenure'][1]),value=int(st.session_state['tenure'][2]))
    vNumOfProducts = st.slider('NumOfProducts',int(st.session_state['NumOfProducts'][0]),int(st.session_state['NumOfProducts'][1]),value=int(st.session_state['NumOfProducts'][2]))
    vHasCrCard = st.radio('HasCrCard',[True,False])
    vIsActiveMember = st.radio('IsActiveMember',[True,False])
    vEstimatedSalary = st.slider('Estimated Salary',st.session_state['EstimatedSalary'][0],st.session_state['EstimatedSalary'][1],value=st.session_state['EstimatedSalary'][2])
    
# Example input data 
input_data = {
    'CreditScore': vCreditScore,
    'Geography' : vGeography, 
    'Gender' : vGender,
    'Age' : vAge,
    'Tenure' : vTenure, 
    'Balance' : vBalance,
    'NumOfProducts' : vNumOfProducts, 
    'HasCrCard' : vHasCrCard, 
    'IsActiveMember' :vIsActiveMember, 
    'EstimatedSalary' : vEstimatedSalary
}

input_df = pd.DataFrame(input_data, index=[0])

st.write('Input:')
st.write(input_df)

input_df['Gender'] = labelEncoder.transform(input_df['Gender'])
input_df = pd.get_dummies(input_df)

for col in st.session_state['columns']:
    if col not in input_df.columns:
        if col not in ( 'Exited', 'Gender_Male','Gender_Female'):
            input_df[col] = False

@st.cache_data
def get_model_coders():
    tf_model = load_model(r"churn_model.h5")

    with open(r'labelEncoder.pkl','rb') as f:
        labelEncoder = pkl.load(f)

    with open(r'standardSacler.pkl','rb') as f:
        standardScaler = pkl.load(f)
    
    return tf_model, labelEncoder , standardScaler

tf_model, labelEncoder , standardScaler = get_model_coders()

# st.write(labelEncoder.transform(['Female']))
input_df['Gender'] = 1 if vGender == 'Male' else 0 

input_df = standardScaler.transform(input_df)

result = tf_model.predict(input_df)

if result >=0.5:
    st.success("The customer won't churn")
else:
    st.error("The customer will churn")    

st.info(f"Churn Score: {result.item():.2f}")
