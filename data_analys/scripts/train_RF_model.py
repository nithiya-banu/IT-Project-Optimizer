#Python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the preprocessed data
print("Loading the datasets...")
projects = pd.read_csv("data/processed_project_data.csv")
clients = pd.read_csv("data/processed_client_data.csv")
print("Datasets loaded successfully!\n")

# Project Delay Prediction (Unchanged)
X_projects = projects.drop(columns=["Delay (Days)"])
y_projects = projects["Delay (Days)"]
X_train_projects, X_test_projects, y_train_projects, y_test_projects = train_test_split(X_projects, y_projects, test_size=0.2, random_state=42)
print("Train-test split for Project Delay completed!\n")

rf_model_projects = RandomForestRegressor(n_estimators=100, random_state=42)
print("Running cross-validation for Project Delay...")
cv_scores_projects = cross_val_score(rf_model_projects, X_projects, y_projects, cv=5, scoring='neg_mean_squared_error')
print(f"\nProject Delay - Cross-validation MSE (5-fold): {cv_scores_projects}")
print(f"Average MSE for Project Delay: {-cv_scores_projects.mean()}\n")

print("Training the Random Forest model for Project Delay prediction...")
rf_model_projects.fit(X_train_projects, y_train_projects)
print("Model training completed for Project Delay!\n")

y_pred_projects_rf = rf_model_projects.predict(X_test_projects)
mse_projects_rf = mean_squared_error(y_test_projects, y_pred_projects_rf)
r2_projects_rf = r2_score(y_test_projects, y_pred_projects_rf)

print("\nProject Delay Prediction Results on Test Set (Random Forest):")
print(f"Mean Squared Error: {mse_projects_rf}")
print(f"R-squared: {r2_projects_rf}\n")

# Client Satisfaction Prediction (Optimized)
X_clients = clients.drop(columns=["Satisfaction Score"])
y_clients = clients["Satisfaction Score"]
X_train_clients, X_test_clients, y_train_clients, y_test_clients = train_test_split(X_clients, y_clients, test_size=0.2, random_state=42)
print("Train-test split for Client Satisfaction completed!\n")

# Hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model_clients = RandomForestRegressor(random_state=42)

print("Tuning Random Forest for Client Satisfaction...")
rf_search = RandomizedSearchCV(
    rf_model_clients, param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42
)
rf_search.fit(X_train_clients, y_train_clients)

best_rf_clients = rf_search.best_estimator_
print(f"\nBest Params: {rf_search.best_params_}")

y_pred_clients_rf = best_rf_clients.predict(X_test_clients)
mse_clients_rf = mean_squared_error(y_test_clients, y_pred_clients_rf)
r2_clients_rf = r2_score(y_test_clients, y_pred_clients_rf)

print("\nTuned Client Satisfaction Prediction Results:")
print(f"Mean Squared Error: {mse_clients_rf}")
print(f"R-squared: {r2_clients_rf}\n")

# Save models
print("Saving the models...")
joblib.dump(rf_model_projects, 'rf_model_project_delay.pkl')
joblib.dump(best_rf_clients, 'rf_model_client_satisfaction.pkl')
print("\nRandom Forest models saved successfully!")
