# data_preparation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_sample_data():
    np.random.seed(42)

    # Creating a sample dataset
    data = pd.DataFrame({
        'grades': np.random.randint(50, 100, 500),
        'attendance': np.random.uniform(0.5, 1.0, 500),
        'age': np.random.randint(17, 25, 500),
        'financial_status': np.random.choice(['lower class', 'middle class', 'wealthy class'], 500),
        'enrolled': np.random.choice([0, 1], 500)
    })

    # Converting categorical variable to dummy variables
    data = pd.get_dummies(data, columns=['financial_status'], drop_first=True)

    return data

def split_data(data):

    # Splitting the data into features and target
    X = data.drop('enrolled', axis=1)
    y = data['enrolled']

    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
