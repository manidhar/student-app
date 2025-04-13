import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import logging

logging.basicConfig(level=logging.INFO)

class StudentPerformancePredictor:
    def __init__(self, model_path, mongo_uri, db_name, collection_name):
        self.model_path = model_path
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MongoClient(self.mongo_uri, server_api=ServerApi('1'))
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.model, self.scalar, self.le = self.load_model()

    def load_model(self):
        logging.info("Loading model from %s", self.model_path)
        with open(self.model_path, 'rb') as file:
            model, scalar, le = pickle.load(file)
        return model, scalar, le

    def preprocessing_input_data(self, data):
        logging.info("Preprocessing input data")
        data['Extracurricular Activities'] = self.le.transform([data['Extracurricular Activities']])[0]
        df = pd.DataFrame([data])
        df_transformed = self.scalar.transform(df)
        return df_transformed

    def predict_data(self, data):
        logging.info("Predicting data")
        processed_data = self.preprocessing_input_data(data)
        prediction = self.model.predict(processed_data)
        return prediction

    def save_prediction(self, data):
        logging.info("Saving prediction to MongoDB")
        self.collection.insert_one(data)

def main():
    predictor = StudentPerformancePredictor(
        model_path='student_lr_final_model.pkl',
        mongo_uri="mongodb+srv://samuthasya:samuthasya@cluster0.g4evw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        db_name='student',
        collection_name='student_perd'
    )

    st.title("Student Performance Prediction")
    st.write("Enter the details of the student to predict the performance")
    hour_studied = st.number_input("Hours Studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("Previous Score", min_value=40, max_value=100, value=70)
    extra = st.selectbox("Extra Curricular Activities", ['Yes', 'No'])
    sleeping_hours = st.number_input("Sleeping Hours", min_value=4, max_value=10, value=7)
    no_of_paper_solved = st.number_input("Number of Question Papers Solved", min_value=0, max_value=10, value=5)

    if st.button("Predict Your Score"):
        user_data = {
            'Hours Studied': hour_studied,
            'Previous Scores': previous_score,
            'Extracurricular Activities': extra,
            'Sleep Hours': sleeping_hours,
            'Sample Question Papers Practiced': no_of_paper_solved
        }
        prediction = predictor.predict_data(user_data)
        st.success(f"Your Predicted Score is {prediction[0]}")
        user_data['Predicted Score'] = round(float(prediction[0]), 2)
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}
        predictor.save_prediction(user_data)

if __name__ == '__main__':
    main()