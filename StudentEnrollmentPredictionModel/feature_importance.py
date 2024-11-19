#Splitting the data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Simulating dataset
np.random.seed(42)
data = pd.DataFrame({
    'grades': np.random.randint(50, 100, 500),
    'attendance': np.random.uniform(0.5, 1.0, 500),
    'age': np.random.randint(17, 25, 500),
    'financial_status': np.random.choice(['lower class', 'middle class', 'wealthy class'], 500),
    'enrolled': np.random.choice([0, 1], 500)
})

# Converting categorical variables to dummy variables
data = pd.get_dummies(data, columns=['financial_status'], drop_first=True)

# Features and target
X = data.drop('enrolled', axis=1)
y = data['enrolled']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#Making predictions
y_pred = model.predict(X_test)

#Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()

#Feature Importance

def plot_feature_importance(model, feature_names):
   plt.figure(figsize=(10, 6))
   plt.barh(X.columns, model.feature_importances_)
   plt.title("Feature Importance in Random Forest Model")
   plt.xlabel("Importance")
   plt.ylabel("Feature")
   plt.show()
