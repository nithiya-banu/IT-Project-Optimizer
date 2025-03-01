#Python 
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest models
rf_model_projects = joblib.load('rf_model_project_delay.pkl')
rf_model_clients = joblib.load('rf_model_client_satisfaction.pkl')

print("Training feature names:", rf_model_clients.feature_names_in_)

def predict_project_delay(input_data): 
    input_df = pd.DataFrame([input_data])
     
    expected_columns_projects = [
        "Bugs Found", "Bugs Fixed", "Bug Fix Ratio", "Team Size", "Hours Worked", 
        "Budget (USD)", "Risk Factor", "Critical Issues Found", "Critical Issue Ratio", 
        "Feature Delivery Ratio", "Team Skill Level", "Resource Utilization Rate", 
        "Client Involvement", "Complexity Score"
    ]
    
    input_df = input_df[expected_columns_projects]
    
    label_encoder = LabelEncoder()
    
    categorical_columns_projects = ["Team Skill Level", "Client Involvement", "Complexity Score"]
    for col in categorical_columns_projects:
        input_df[col] = label_encoder.fit_transform(input_df[col])

    delay_prediction = rf_model_projects.predict(input_df)
    return delay_prediction[0]

def predict_client_satisfaction(input_data):
    input_df = pd.DataFrame([input_data])
    
    expected_columns_clients = ['Complaints Raised', 'Complaints Resolved', 'Complaint Resolution Ratio',
    'Resolution Time (hours)','Feedback Sentiment' ,'Revenue Generated (USD)',
    'Service Renewal Likelihood (%)', 'Client Communication Rating',
    'Past Renewals' ,'Churn Risk', 'Account Manager Engagement',
    'Product Usability Score', 'Customer Tenure']
    
    input_df = input_df[expected_columns_clients]
    
    label_encoder = LabelEncoder()
    
    categorical_columns_clients = ["Feedback Sentiment", "Churn Risk"]
    for col in categorical_columns_clients:
        input_df[col] = label_encoder.fit_transform(input_df[col])

    # Make prediction
    satisfaction_prediction = rf_model_clients.predict(input_df)
    return satisfaction_prediction[0]

project_input_data = {
    "Bugs Found": 50,
    "Bugs Fixed": 45,
    "Bug Fix Ratio": 0.9,
    "Team Size": 15,
    "Hours Worked": 1200,
    "Budget (USD)": 100000,
    "Risk Factor": 5,
    "Critical Issues Found": 5,
    "Critical Issue Ratio": 0.1,
    "Feature Delivery Ratio": 0.93,
    "Team Skill Level": "High",
    "Resource Utilization Rate": 85,
    "Client Involvement": "Medium",
    "Complexity Score": "Medium"
}

client_input_data = {
    "Complaints Raised": 3,
    "Complaints Resolved": 3,
    "Complaint Resolution Ratio": 1.0,
    "Resolution Time (hours)": 4,
    "Feedback Sentiment": "Positive",
    "Revenue Generated (USD)": 50000,
    "Service Renewal Likelihood (%)": 80,
    "Client Communication Rating": 9,
    "Past Renewals": 2,
    "Churn Risk": "Low",
    "Account Manager Engagement": 7,
    "Product Usability Score": 8,
    "Customer Tenure": 5
}

# Make predictions
project_delay = predict_project_delay(project_input_data)
client_satisfaction = predict_client_satisfaction(client_input_data)

# Display the predictions
print(f"Predicted Project Delay (Days): {project_delay}")
print(f"Predicted Client Satisfaction Score: {client_satisfaction}")
