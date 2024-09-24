# Loan-Default-Risk-Analysis-End-to-End


# Loan Default Risk Prediction Project

This project aims to predict loan default risk based on various demographic and financial attributes of applicants. Using a dataset of 100,000 records, the model assesses risk using machine learning techniques, providing insights into factors that contribute to default risk.

## Project Overview

### Dataset
The dataset consists of 100,000 records with the following columns:
- **Applicant_ID**: Unique identifier for each applicant
- **Annual_Income**: Income of the applicant in local currency
- **Applicant_Age**: Age of the applicant
- **Work_Experience**: Years of work experience
- **Marital_Status**: Marital status of the applicant (Single, Married, etc.)
- **House_Ownership**: Ownership of a house (Owned, Rented, etc.)
- **Vehicle_Ownership**: Ownership of a vehicle
- **Occupation**: Occupation of the applicant
- **Residence_City**: City of residence
- **Residence_State**: State of residence
- **Years_in_Current_Employment**: Duration of employment in current job
- **Years_in_Current_Residence**: Duration of stay in the current residence
- **Loan_Default_Risk**: Target variable indicating the risk of default

### Files Included
- **Loan Dataset Analysis.ipynb**: Jupyter Notebook with the analysis and model building process.
- **app.py**: Python script to serve the model as a web API.
- **best_random_forest_model.pkl**: Pre-trained Random Forest model.
- **scaler.pkl**: StandardScaler model to scale the input features.
- **expected_columns.pkl**: Expected column names for the model.

## Key Features
- **Data Preprocessing**: The dataset undergoes preprocessing, including missing value handling, feature scaling, and encoding categorical variables.
- **Model Training**: A Random Forest Classifier is used to predict loan default risk, selected after comparing multiple models.
- **Web Application**: A Python Flask web app is provided to allow users to make predictions on new data by loading the trained model.

## How to Run the Project

### Prerequisites
- Python 3.8 or higher
- `Flask` for serving the web application
- `scikit-learn` for machine learning operations
- `pandas` and `numpy` for data processing

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-default-risk-prediction.git
   cd loan-default-risk-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Application
1. Ensure the `app.py`, `best_random_forest_model.pkl`, `scaler.pkl`, and `expected_columns.pkl` files are in the same directory.
2. Start the Flask app:
   ```bash
   python app.py
   ```

3. Access the web application by navigating to `http://127.0.0.1:5000/` in your browser.

### Making Predictions
To make predictions, you can either use the form provided in the web app interface or send a POST request with JSON data to the endpoint:
```
POST http://127.0.0.1:5000/predict
```
The JSON format should match the expected schema defined in the `expected_columns.pkl` file.

### Example Input:
```json
{
  "Annual_Income": 50000,
  "Applicant_Age": 35,
  "Work_Experience": 10,
  "Marital_Status": "Single",
  "House_Ownership": "Owned",
  "Vehicle_Ownership": "No",
  "Occupation": "Engineer",
  "Residence_City": "Toronto",
  "Residence_State": "ON",
  "Years_in_Current_Employment": 5,
  "Years_in_Current_Residence": 3
}
```

### Example Response:
```json
{
  "Loan_Default_Risk": "Low"
}
```

## Model Evaluation
The Random Forest model was evaluated using cross-validation, achieving the following metrics:

| Model           | Accuracy | Precision | Recall   | F1-Score | AUC Score |
|-----------------|----------|-----------|----------|----------|-----------|
| Random Forest   | 0.92935  | 0.689942  | 0.828780 | 0.753015 | 0.970297  |

Further improvements can be made by experimenting with different feature engineering techniques and algorithms.

## Future Enhancements
- Implement additional model improvements such as hyperparameter tuning or alternative algorithms.
- Add real-time data collection for more dynamic predictions.
- Build a front-end interface with enhanced user experience.

## Conclusion
This project provides a full end-to-end solution for predicting loan default risk, from data analysis and model building to deploying the model as a web service. It demonstrates the practical application of machine learning in financial risk assessment.
