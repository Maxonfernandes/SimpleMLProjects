import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
diabetes_ds = pd.read_csv('/Users/maxonfernandes/Desktop/Beginner ML Projects/diabetes predictor/diabetes.csv')

# Split the dataset into features (x) and target (y)
x = diabetes_ds.drop('Outcome', axis=1)
y = diabetes_ds['Outcome']

# Data Standardization
scaler = StandardScaler()
scaler.fit(x)
standarddata = scaler.transform(x)

# Update 'x' with the standardized data
x = standarddata

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Training the SVM classifier with a linear kernel
classifier = SVC(kernel='linear')
classifier.fit(xtrain, ytrain)

# Streamlit app
st.title("Diabetes Prediction App")
st.write('Enter The Patient Details:')
st.write('1.Pregnancies  2.Glucose  3.BloodPressure  4.SkinThickness  5.Insulin  6.BMI  7.DiabetesPedigreeFunction  8.Age')
input_data = st.text_input("\n(e.g., 3,126,88,41,235,39.3,0.704,27):")

if st.button("Predict"):
    # Convert the input data to a NumPy array
    input_data_as_numpy_array = np.array([float(x) for x in input_data.split(',')])

    # Reshape the array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    # Make the prediction
    prediction = classifier.predict(std_data)

    if prediction == 0:
        st.write('THE PATIENT IS NON-DIABETIC')
    else:
        st.write('THE PATIENT IS DIABETIC')
