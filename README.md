# Optimising Client Satisfaction and IT Project Delivery

This project focuses on predicting IT project delays and client satisfaction using synthetic data and machine learning. The complete workflow includes data generation, preprocessing, model training with hyperparameter tuning, prediction, and visualization.

---

## Overview

- **Data Generation:**  
  Synthetic datasets for IT projects and clients are generated using Faker and random data. Project data includes metrics such as bugs found, team size, and resource utilization. Client data captures details like complaints, resolution times, revenue, and satisfaction scores.

- **Data Preprocessing:**  
  Data cleaning, label encoding for categorical variables, and feature scaling are performed. The datasets are then split into training and testing sets.

- **Model Training:**  
  Two Random Forest models are developed:
  - **Project Delay Prediction Model:** Predicts project delay (in days) based on various project parameters.
  - **Client Satisfaction Prediction Model:** Predicts client satisfaction scores with hyperparameter tuning using RandomizedSearchCV.

- **Prediction:**  
  The trained models are loaded to make predictions on new data inputs. Utility functions preprocess the inputs and output predictions for project delays and client satisfaction.

- **Visualization:**  
  Multiple visualizations are generated to analyze the data and model performance, including:
  - Distribution of project delays.
  - Relationship between complaints resolved and satisfaction score.
  - Churn risk distribution.
  - Correlation heatmap for client data.
  - Budget vs. delay comparison.

---
## Getting Started
### Prerequisites

- Python 3.x  
- A virtual environment is recommended.

**Required Packages:**  
All dependencies are listed in the `requirements.txt` file.  
Install the required packages using:

```bash
pip install -r requirements.txt
```

### Note for Linux Users:

Ensure that the python3-tk package is installed for visualization support:
    
```bash
    sudo apt-get install python3-tk
```

### Project Structure

```bash
.
├── data_generation.py        # Generates and preprocesses synthetic datasets
├── train_RF_model.py         # Trains Random Forest models for predictions
├── Load_RF_model.py          # Loads trained models and makes predictions on input data
├── visualize.py              # Generates and saves visualization plots
├── requirements.txt          # List of required Python packages
├── README.md                 # Project overview and setup instructions
├── web                       # Contains additional files (e.g., bugs.txt for Linux tk installation)
└── virtual                   # Virtual environment folder
```

### Running the Project

#### Data Generation and Preprocessing:
- Run data_generation.py to create processed CSV files for projects and clients.

```bash
python data_generation.py
 ```
#### Model Training:
Train the Random Forest models by running train_RF_model.py. This script will save the trained models as pickle files.

```bash 
python train_RF_model.py
```

#### Make Predictions:
Use Load_RF_model.py to load the saved models and run predictions on sample input data.


```bash
python Load_RF_model.py
```

#### Visualization:
Generate visualizations by executing visualize.py. The plots will be saved in the plots directory.

```bash
python visualize.py
```

## Results
### Project Delay Prediction:
Evaluated using metrics like Mean Squared Error and R-squared on the test set.

### Client Satisfaction Prediction:
Model performance is enhanced via hyperparameter tuning and evaluated similarly with performance metrics.

### Visualizations:
A set of plots is generated to provide insights into project delays, client satisfaction, and other key metrics.

## Download Pre-Trained Model
To use the trained model, download it from [Google Drive]([YOUR_LINK_HERE](https://drive.google.com/drive/folders/1MleIBVlktlS4uc4VywlNvEfg_k-1Rsbw?usp=sharing)) and place it in the project directory.

