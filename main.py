import streamlit as st
import pickle
import pandas as pd
import numpy as np

@st.cache(persist= True)
def load_data():
    df = pd.read_csv('Placement_Data_Full_Class.csv')
    df.drop(['sl_no'], axis= 1, inplace= True)
    return df

def user_input():
    st.sidebar.subheader("Hyper Parameter Tuning")
    gender = st.sidebar.radio("Select the Gender", ('M', 'F'))
    ssc_percent = st.sidebar.slider('Select your SSC Score (in %)', 40, 100, 50)
    hsc_percent = st.sidebar.slider("Select your HSC Score (in %)", 40, 100, 50)
    degree_percent = st.sidebar.slider("Select your Degree Score (in %)", 40, 100, 50)
    work_exp = st.sidebar.radio('Do you have any Work Experience?', ('Yes', 'No'))
    employ_test_percent = st.sidebar.slider("Select the Test Score taken by Employer (in %)", 40, 100, 50)
    specialisation = st.sidebar.selectbox("Select your Specialization in the course", ['Mkt&HR', 'Mkt&Fin'])
    mba_percent = st.sidebar.slider("Select your Score in MBA (in %)", 40, 100, 50)
    data = {
        'Gender': gender, 
        'SSC_Percent': ssc_percent, 
        'HSC_Percent': hsc_percent, 
        'Degree_Percent': degree_percent, 
        'WorkEx': work_exp, 
        'Employability-Test_Scores': employ_test_percent, 
        'Specialisation': specialisation, 
        'MBA_Percent': mba_percent}
    # Creating the dataframe
    df_user = pd.DataFrame(data, index= [0])
    return df_user

def preprocessing(df_user):
    df_user['Gender'].replace(['M', 'F'], [0, 1], inplace= True)
    df_user['WorkEx'].replace(['Yes', 'No'], [1, 0], inplace= True)
    df_user['Specialisation'].replace(['Mkt&HR', 'Mkt&Fin'], [0, 1], inplace= True)
    return df_user

def load_pickle(algo):
    if algo == 'Logistic Regression Classifier':
        return pickle.load(open('lr_model.pkl', 'rb'))
    if algo == 'Random Forest Classifier':
        return pickle.load(open('rfc_model.pkl', 'rb'))
    if algo == 'Support Vector Machine Classifier':
        return pickle.load(open('svc_model.pkl', 'rb'))
    if algo == 'Gaussian Naïve Bayes Classifier':
        return pickle.load(open('gnb_model.pkl', 'rb'))
     
def prediction(df_user, pkl_file):
    X = df_user.iloc[:, :8].values
    prediction = pkl_file.predict(X)
    prediction_prob = pkl_file.predict_proba(X)
    if prediction == 0:
        st.write("You'll not be placed!")
    else:
        st.write("You'll be placed!")

if __name__ == '__main__':
    st.set_page_config(page_title= 'Placement Prediction', layout= 'wide')
    st.header("Placement Prediction App 🎓")
    st.subheader("App to predict if student is placed or not!")
    if st.checkbox('Check dataset'):
        df = load_data()
        st.dataframe(df)
    
    df_user = user_input()
    if st.checkbox("Check your input"):
        st.write(df_user)
    df_u = preprocessing(df_user)
    # Select the algorithm
    algo = st.sidebar.selectbox('Select the Algorithm to apply', ['Logistic Regression Classifier', 'Random Forest Classifier', 'Support Vector Machine Classifier', 'Gaussian Naïve Bayes Classifier'])
    pkl_file = load_pickle(algo)
    # Predict 
    if st.button('Predict'):
        prediction(df_u, pkl_file)