import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
from tqdm import tqdm  # Import tqdm for progress bar

# Load the preprocessed data
print("Loading the datasets...")
projects = pd.read_csv("data/processed_project_data.csv")
clients = pd.read_csv("data/processed_client_data.csv")
print("Datasets loaded successfully!\n")

# Split the data for Project Delay Prediction
X_projects = projects.drop(columns=["Delay (Days)"])
y_projects = projects["Delay (Days)"]

# Train-test split for project data (80% train, 20% test)
X_train_projects, X_test_projects, y_train_projects, y_test_projects = train_test_split(X_projects, y_projects, test_size=0.2, random_state=42)
print("Train-test split for Project Delay completed!\n")

# Replace Linear Regression with Random Forest for Project Delay Prediction
# Random Forest Model
rf_model_projects = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross-validation for Project Delay Prediction (5-fold)
print("Running cross-validation for Project Delay...")
cv_scores_projects = []
for _ in tqdm(range(5), desc="Cross-Validating Project Delay"):
    score = cross_val_score(rf_model_projects, X_projects, y_projects, cv=5, scoring='neg_mean_squared_error')
    cv_scores_projects.append(score)

# Evaluate cross-validation results
print(f"\nProject Delay - Cross-validation MSE (5-fold): {cv_scores_projects}")
print(f"Average MSE for Project Delay: {-sum(cv_scores_projects)/len(cv_scores_projects)}\n")

# Train the Random Forest model on the entire training data
print("Training the Random Forest model for Project Delay prediction...")
rf_model_projects.fit(X_train_projects, y_train_projects)
print("Model training completed for Project Delay!\n")

# Make predictions on the test set
print("Making predictions on the Project Delay test set...")
y_pred_projects_rf = rf_model_projects.predict(X_test_projects)

# Evaluate the model on the test set
mse_projects_rf = mean_squared_error(y_test_projects, y_pred_projects_rf)
r2_projects_rf = r2_score(y_test_projects, y_pred_projects_rf)

print("\nProject Delay Prediction Results on Test Set (Random Forest):")
print(f"Mean Squared Error: {mse_projects_rf}")
print(f"R-squared: {r2_projects_rf}\n")

# --- Repeat for Client Satisfaction ---

# Split the data for Client Satisfaction Prediction
X_clients = clients.drop(columns=["Satisfaction Score"])
y_clients = clients["Satisfaction Score"]

# Train-test split for client data (80% train, 20% test)
X_train_clients, X_test_clients, y_train_clients, y_test_clients = train_test_split(X_clients, y_clients, test_size=0.2, random_state=42)
print("Train-test split for Client Satisfaction completed!\n")

# Random Forest Model for Client Satisfaction
rf_model_clients = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross-validation for Client Satisfaction Prediction (5-fold)
print("Running cross-validation for Client Satisfaction...")
cv_scores_clients = []
for _ in tqdm(range(5), desc="Cross-Validating Client Satisfaction"):
    score = cross_val_score(rf_model_clients, X_clients, y_clients, cv=5, scoring='neg_mean_squared_error')
    cv_scores_clients.append(score)

# Evaluate cross-validation results
print(f"\nClient Satisfaction - Cross-validation MSE (5-fold): {cv_scores_clients}")
print(f"Average MSE for Client Satisfaction: {-sum(cv_scores_clients)/len(cv_scores_clients)}\n")

# Train the Random Forest model on the entire training data
print("Training the Random Forest model for Client Satisfaction prediction...")
rf_model_clients.fit(X_train_clients, y_train_clients)
print("Model training completed for Client Satisfaction!\n")

# Make predictions on the test set
print("Making predictions on the Client Satisfaction test set...")
y_pred_clients_rf = rf_model_clients.predict(X_test_clients)

# Evaluate the model on the test set
mse_clients_rf = mean_squared_error(y_test_clients, y_pred_clients_rf)
r2_clients_rf = r2_score(y_test_clients, y_pred_clients_rf)

print("\nClient Satisfaction Prediction Results on Test Set (Random Forest):")
print(f"Mean Squared Error: {mse_clients_rf}")
print(f"R-squared: {r2_clients_rf}\n")

# Save the Random Forest models (optional, if you want to reuse the models later)
print("Saving the models...")
joblib.dump(rf_model_projects, 'rf_model_project_delay.pkl')
joblib.dump(rf_model_clients, 'rf_model_client_satisfaction.pkl')
print("\nRandom Forest models saved successfully!")

# ---- Now for Linear Regression ----

# Linear Regression for Project Delay
lr_model_projects = LinearRegression()
print("Training the Linear Regression model for Project Delay prediction...")
lr_model_projects.fit(X_train_projects, y_train_projects)
print("Model training completed for Project Delay with Linear Regression!\n")

# Make predictions
y_pred_projects_lr = lr_model_projects.predict(X_test_projects)

# Evaluate Linear Regression Model for Project Delay
mse_projects_lr = mean_squared_error(y_test_projects, y_pred_projects_lr)
r2_projects_lr = r2_score(y_test_projects, y_pred_projects_lr)

print("\nProject Delay Prediction Results on Test Set (Linear Regression):")
print(f"Mean Squared Error: {mse_projects_lr}")
print(f"R-squared: {r2_projects_lr}\n")

# Linear Regression for Client Satisfaction
lr_model_clients = LinearRegression()
print("Training the Linear Regression model for Client Satisfaction prediction...")
lr_model_clients.fit(X_train_clients, y_train_clients)
print("Model training completed for Client Satisfaction with Linear Regression!\n")

# Make predictions
y_pred_clients_lr = lr_model_clients.predict(X_test_clients)

# Evaluate Linear Regression Model for Client Satisfaction
mse_clients_lr = mean_squared_error(y_test_clients, y_pred_clients_lr)
r2_clients_lr = r2_score(y_test_clients, y_pred_clients_lr)

print("\nClient Satisfaction Prediction Results on Test Set (Linear Regression):")
print(f"Mean Squared Error: {mse_clients_lr}")
print(f"R-squared: {r2_clients_lr}\n")

# Save the Linear Regression models (optional)
print("Saving the models...")
joblib.dump(lr_model_projects, 'lr_model_project_delay.pkl')
joblib.dump(lr_model_clients, 'lr_model_client_satisfaction.pkl')
print("\nLinear Regression models saved successfully!")
