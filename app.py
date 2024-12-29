import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Manually define the column names from your training data
trained_columns = [
    'id', 'Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 
    'Sleep Duration', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 
    'Family History of Mental Illness', 'City_Agra', 'City_Ahmedabad', 'City_Bangalore', 'City_Bhavna', 'City_Bhopal', 
    'City_Chennai', 'City_City', 'City_Delhi', 'City_Faridabad', 'City_Gaurav', 'City_Ghaziabad', 'City_Harsh', 
    'City_Harsha', 'City_Hyderabad', 'City_Indore', 'City_Jaipur', 'City_Kalyan', 'City_Kanpur', 'City_Khaziabad', 
    'City_Kibara', 'City_Kolkata', 'City_Less Delhi', 'City_Less than 5 Kalyan', 'City_Lucknow', 'City_Ludhiana', 
    'City_M.Com', 'City_M.Tech', 'City_ME', 'City_Meerut', 'City_Mihir', 'City_Mira', 'City_Mumbai', 'City_Nagpur', 
    'City_Nalini', 'City_Nalyan', 'City_Nandini', 'City_Nashik', 'City_Patna', 'City_Pune', 'City_Rajkot', 'City_Rashi', 
    'City_Reyansh', 'City_Saanvi', 'City_Srinagar', 'City_Surat', 'City_Thane', 'City_Vaanya', 'City_Vadodara', 
    'City_Varanasi', 'City_Vasai-Virar', 'City_Visakhapatnam', 'Profession_Chef', 'Profession_Civil Engineer', 
    'Profession_Content Writer', 'Profession_Digital Marketer', 'Profession_Doctor', 'Profession_Educational Consultant', 
    'Profession_Entrepreneur', 'Profession_Lawyer', 'Profession_Manager', 'Profession_Pharmacist', 'Profession_Student', 
    'Profession_Teacher', 'Profession_UX/UI Designer', 'Dietary Habits_Moderate', 'Dietary Habits_Others', 
    'Dietary Habits_Unhealthy', 'Degree_B.Com', 'Degree_B.Ed', 'Degree_B.Pharm', 'Degree_B.Tech', 'Degree_BA', 'Degree_BBA', 
    'Degree_BCA', 'Degree_BE', 'Degree_BHM', 'Degree_BSc', 'Degree_Class 12', 'Degree_LLB', 'Degree_LLM', 'Degree_M.Com', 
    'Degree_M.Ed', 'Degree_M.Pharm', 'Degree_M.Tech', 'Degree_MA', 'Degree_MBA', 'Degree_MBBS', 'Degree_MCA', 'Degree_MD', 
    'Degree_ME', 'Degree_MHM', 'Degree_MSc', 'Degree_Others', 'Degree_PhD'
]

# Function to preprocess the input data
def preprocess_data(input_data):
    # Encoding categorical features
    sleep_duration_map = {
        'Less than 5 hours': 1,
        '5-6 hours': 2,
        '7-8 hours': 3,
        'More than 8 hours': 4
    }

    # Dictionary for encoding binary categorical variables
    label_encoder_dict = {
        'Gender': {'Male': 1, 'Female': 0},
        'Have you ever had suicidal thoughts ?': {'Yes': 1, 'No': 0},
        'Family History of Mental Illness': {'Yes': 1, 'No': 0}
    }

    # Encoding categorical variables by extracting the first value (since we only have one entry)
    input_data['Gender'] = label_encoder_dict['Gender'][input_data['Gender'][0]]
    input_data['Have you ever had suicidal thoughts ?'] = label_encoder_dict['Have you ever had suicidal thoughts ?'][input_data['Have you ever had suicidal thoughts ?'][0]]
    input_data['Family History of Mental Illness'] = label_encoder_dict['Family History of Mental Illness'][input_data['Family History of Mental Illness'][0]]
    input_data['Sleep Duration'] = sleep_duration_map[input_data['Sleep Duration'][0]]

    # One-hot encoding for categorical variables
    input_data = pd.get_dummies(input_data, columns=['City', 'Profession', 'Dietary Habits', 'Degree'], drop_first=True)

    # Scaling the numerical features
    scaler = StandardScaler()
    numerical_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 
                          'Work/Study Hours', 'Financial Stress']
    input_data[numerical_features] = scaler.fit_transform(input_data[numerical_features])

    # Ensure that the input data has the same columns as the training data
    input_data = input_data.reindex(columns=trained_columns, fill_value=0)

    return input_data

# Function to make a prediction
def make_prediction(input_data):
    # Preprocess the input data
    processed_data = preprocess_data(input_data)
    
    # Predict using the trained model
    prediction = model.predict(processed_data)
    
    # Return the prediction result
    return prediction

# Streamlit interface
st.title("Student Mental Health Prediction")
st.write("Enter the details below to predict if the student is depressed or not.")

# Get user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=22)
academic_pressure = st.slider("Academic Pressure", 1, 10, value=5)
work_pressure = st.slider("Work Pressure", 1, 10, value=4)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5)
study_satisfaction = st.slider("Study Satisfaction", 1, 5, value=3)
job_satisfaction = st.slider("Job Satisfaction", 1, 5, value=4)
sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=6)
financial_stress = st.slider("Financial Stress", 1, 5, value=3)
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai"])  # Add more cities as needed
profession = st.selectbox("Profession", ["Student", "Doctor", "Engineer", "Teacher"])  # Add more professions
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy", "Others"])
degree = st.selectbox("Degree", ["B.Tech", "B.Sc", "M.Tech", "MCA", "B.Com", "MA", "MSc"])  # Add more degrees

# Prepare input data as a DataFrame
input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Academic Pressure': [academic_pressure],
    'Work Pressure': [work_pressure],
    'CGPA': [cgpa],
    'Study Satisfaction': [study_satisfaction],
    'Job Satisfaction': [job_satisfaction],
    'Sleep Duration': [sleep_duration],
    'Have you ever had suicidal thoughts ?': [suicidal_thoughts],
    'Work/Study Hours': [work_study_hours],
    'Financial Stress': [financial_stress],
    'Family History of Mental Illness': [family_history],
    'City': [city],
    'Profession': [profession],
    'Dietary Habits': [dietary_habits],
    'Degree': [degree]
})

# Make prediction
if st.button("Predict Depression Status"):
    prediction = make_prediction(input_data)
    st.write(f"Prediction: {'Depressed' if prediction[0] == 1 else 'Not Depressed'}")
