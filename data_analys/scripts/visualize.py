#Python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, shutil

if os.path.exists("plots"):
    shutil.rmtree("plots")
os.mkdir("plots/")

# Load the processed datasets
projects = pd.read_csv(r"data_analys\data\processed_project_data.csv")
clients = pd.read_csv(r"data_analys\data\processed_client_data.csv")

sns.set(style="whitegrid")

# 1. Distribution of Project Delays
plt.figure(figsize=(10, 5))
sns.histplot(projects["Delay (Days)"], bins=30, kde=True, color='blue')
plt.title("Distribution of Project Delays")
plt.xlabel("Delay (Days)")
plt.ylabel("Frequency")
plt.savefig("plots/distribution_of_project_delays.png")
plt.close()

# 2. Complaints Resolved vs Satisfaction Score
plt.figure(figsize=(10, 5))
sns.scatterplot(x=clients["Complaints Resolved"], y=clients["Satisfaction Score"], alpha=0.6)
plt.title("Complaints Resolved vs Satisfaction Score")
plt.xlabel("Complaints Resolved")
plt.ylabel("Satisfaction Score")
plt.savefig("plots/complaints_resolved_vs_satisfaction_score.png")
plt.close()

# 3. Churn Risk Distribution
plt.figure(figsize=(7, 5))
sns.countplot(x=clients["Churn Risk"], palette="coolwarm", hue=clients["Churn Risk"], legend=False)
plt.title("Churn Risk Distribution")
plt.xlabel("Churn Risk Level")
plt.ylabel("Count")
plt.savefig("plots/churn_risk_distribution.png")
plt.close()

# 4. Correlation Heatmap for Clients Data
plt.figure(figsize=(12, 6))
sns.heatmap(clients.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Client Data")
plt.savefig("plots/correlation_heatmap_client_data.png")
plt.close()

# 5. Budget vs Delay Comparison
plt.figure(figsize=(10, 5))
sns.scatterplot(x=projects["Budget (USD)"], y=projects["Delay (Days)"], alpha=0.5)
plt.title("Budget vs Delay Comparison")
plt.xlabel("Budget (USD)")
plt.ylabel("Delay (Days)")
plt.savefig("plots/budget_vs_delay_comparison.png")
plt.close()

print("Visualizations generated successfully!")
