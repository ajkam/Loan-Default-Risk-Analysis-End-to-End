# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sklearn


# Load the trained model and scaler
model = joblib.load('best_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('expected_columns.pkl')

# Load dataset to extract unique values for categorical inputs
applicant_details = pd.read_csv(r'C:\Users\Annuj\Downloads\archive (11)\Applicant-details.csv')

# Extract unique options for cities, states, and occupations
unique_cities = sorted(applicant_details['Residence_City'].unique())
unique_states = sorted(applicant_details['Residence_State'].unique())
unique_occupations = sorted(applicant_details['Occupation'].unique())

# Define the preprocessing function
def preprocess_input(df):
    # 1. Data Type Conversion
    # Ensure numerical columns are of type float
    numerical_cols = ['Annual_Income', 'Applicant_Age', 'Work_Experience',
                      'Years_in_Current_Employment', 'Years_in_Current_Residence']
    df[numerical_cols] = df[numerical_cols].astype(float)
    
    # 2. Encode categorical variables
    # Label Encoding for binary categorical variables
    binary_cols = ['Marital_Status', 'House_Ownership', 'Vehicle_Ownership(car)']
    for col in binary_cols:
        if col == 'Marital_Status':
            df[col] = df[col].map({'Single': 0, 'Married': 1})
        elif col == 'House_Ownership':
            df[col] = df[col].map({'Rent': 0, 'Own': 1, 'Mortgage': 2})
        elif col == 'Vehicle_Ownership(car)':
            df[col] = df[col].map({'No': 0, 'Yes': 1})
    
    # One-Hot Encoding for 'Occupation'
    # Assuming you used get_dummies during training
    occupation_dummies = pd.get_dummies(df['Occupation'], prefix='Occupation')
    df = pd.concat([df, occupation_dummies], axis=1)
    df.drop('Occupation', axis=1, inplace=True)
    
    # One-Hot Encoding for 'Residence_City' and 'Residence_State'
    city_dummies = pd.get_dummies(df['Residence_City'], prefix='Residence_City')
    df = pd.concat([df, city_dummies], axis=1)
    df.drop('Residence_City', axis=1, inplace=True)
    
    state_dummies = pd.get_dummies(df['Residence_State'], prefix='Residence_State')
    df = pd.concat([df, state_dummies], axis=1)
    df.drop('Residence_State', axis=1, inplace=True)
    
   # 3. Feature Engineering
    # Compute engineered features
    df['Employment_Stability'] = df['Years_in_Current_Employment'] / df['Applicant_Age']
    df['Residence_Stability'] = df['Years_in_Current_Residence'] / df['Applicant_Age']
    df['Income_per_Age'] = df['Annual_Income'] / df['Applicant_Age']
    df['Work_Experience_Ratio'] = df['Work_Experience'] / df['Applicant_Age']
    df['Income_Stability'] = df['Years_in_Current_Employment'] / df['Work_Experience']
    df['Asset_Ownership_Score'] = df['House_Ownership'] + df['Vehicle_Ownership(car)']
    df['Tenure_Ratio'] = df['Years_in_Current_Residence'] / df['Years_in_Current_Employment']
    df['Employment_Gap'] = df['Applicant_Age'] - df['Work_Experience']
    
   # Handle infinite and NaN values
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    
     # 4. Scaling Engineered Numerical Features
    # Retrieve features the scaler was fitted on
    scaler_features = scaler.feature_names_in_

    # Ensure these features are in the DataFrame
    missing_features = [feat for feat in scaler_features if feat not in df.columns]
    if missing_features:
        for feat in missing_features:
            df[feat] = 0  # or any appropriate default value

    # Apply the scaler to these features
    df[scaler_features] = scaler.transform(df[scaler_features])

    # 5. Ensure All Expected Columns Are Present
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training data
    df = df[expected_columns]

    return df
    
    # 5. Ensure all expected columns are present
      
    # Add missing columns with zeros
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[expected_columns]
    
    return df

# Define the main function
def main():
    st.title("Loan Default Prediction App")
    st.write("Enter the applicant's information to predict loan default risk.")
    
    # Collect user input for each feature
    # For numerical features, use sliders or number inputs
    annual_income = st.slider('Annual Income', min_value=0, max_value=200000, value=50000, step=1000)
    applicant_age = st.slider('Applicant Age', min_value=18, max_value=100, value=30, step=1)
    work_experience = st.slider('Work Experience (years)', min_value=0, max_value=50, value=5, step=1)
    years_in_current_employment = st.slider('Years in Current Employment', min_value=0, max_value=50, value=2, step=1)
    years_in_current_residence = st.slider('Years in Current Residence', min_value=0, max_value=50, value=3, step=1)
    
    # For categorical features, use select boxes with dynamic options
    marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
    house_ownership = st.selectbox('House Ownership', ['Rent', 'Own', 'Mortgage'])
    vehicle_ownership = st.selectbox('Vehicle Ownership(car)', ['No', 'Yes'])
    occupation = st.selectbox('Occupation', unique_occupations)
    residence_city = st.selectbox('Residence City', unique_cities)
    residence_state = st.selectbox('Residence State', unique_states)
    
    # When the user clicks the 'Predict' button
    if st.button('Predict'):
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'Annual_Income': [annual_income],
            'Applicant_Age': [applicant_age],
            'Work_Experience': [work_experience],
            'Years_in_Current_Employment': [years_in_current_employment],
            'Years_in_Current_Residence': [years_in_current_residence],
            'Marital_Status': [marital_status],
            'House_Ownership': [house_ownership],
            'Vehicle_Ownership(car)': [vehicle_ownership],
            'Occupation': [occupation],
            'Residence_City': [residence_city],
            'Residence_State': [residence_state]
        })
        
        # Preprocess the input data to match the training data
        input_data_processed = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_processed)
        prediction_proba = model.predict_proba(input_data_processed)[:, 1]
        
        # Display the result
        if prediction[0] == 1:
            st.error(f"Prediction: The loan is **likely to default**. (Probability of default: {prediction_proba[0]:.2%})")
        else:
            st.success(f"Prediction: The loan is **not likely to default**. (Probability of default: {prediction_proba[0]:.2%})")

if __name__ == '__main__':
    main()
