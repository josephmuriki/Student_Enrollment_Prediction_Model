# main.py
from data_preparation import create_sample_data, split_data
from model_training import train_model
from model_evaluation import model_evaluation
from feature_importance import plot_feature_importance
from model_saving import model_saving
from predictions_export import export_predictions

#Creating a sample dataset
data = create_sample_data()

#Splitting the data
X_train, X_test, y_train, y_test = split_data(data)

#Training the model
model = train_model(X_train, y_train)

#Evaluating the model
accuracy, conf_matrix, class_report = model_evaluation(model, X_test, y_test)
print(f"Model Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

#Feature importance
plot_feature_importance(model, X_train)

#Saving the model
model_saving(model)

#Exporting predictions
y_pred = model.predict(X_test)
export_predictions(X_test, y_test, y_pred)
