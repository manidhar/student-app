import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://samuthasya:samuthasya@cluster0.g4evw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db=client['student']
collection=db['student_perd']

def load_model():
    with open('student_lr_final_model.pkl', 'rb') as file:
        model,scalar,le = pickle.load(file)
    return model,scalar,le
    
def preprocessing_input_data(data,scalar,le):
    data['Extracurricular Activities']= le.transform([data['Extracurricular Activities']])[0]
    df=pd.DataFrame([data])
    df_tranformed = scalar.transform(df)
    return df_tranformed

def predict_data(data):
    model,scalar,le = load_model()
    processed_data = preprocessing_input_data(data,scalar,le)
    prediction = model.predict(processed_data)
    return prediction


def main():
    st.title("Student Performance Prediction")
    st.write("Enter the details of the student to predict the performance")
    hour_studied=st.number_input("Hours Studied",min_value=1,max_value=10,value=5)
    previous_score=st.number_input("Previous Score",min_value=40,max_value=100,value=70)
    extra=st.selectbox("Extra Curricular Activities",['Yes','No'])
    sleeping_hours=st.number_input("slepping Hours",min_value=4,max_value=10,value=7)
    no_of_paper_solved=st.number_input("number of question papers solved",min_value=0,max_value=10,value=5)

    if st.button("predict-your_score"):
        user_data = {
            'Hours Studied':hour_studied,
            'Previous Scores':previous_score,
            'Extracurricular Activities':extra,
            'Sleep Hours':sleeping_hours,
            'Sample Question Papers Practiced':no_of_paper_solved
            }
        prediction = predict_data(user_data)
        st.success(f"Your Predicted Score is {prediction}")
        user_data['Predicted Score'] = round(float(prediction[0]),2)
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}
        collection.insert_one(user_data)


if __name__ == '__main__':
    main()