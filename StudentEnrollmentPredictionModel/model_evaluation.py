# model_evaluation.py
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def model_evaluation(model, X_test, y_test):
    # Making predictions
    y_pred = model.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generating confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Classification report
    class_report = classification_report(y_test, y_pred)

    return accuracy, conf_matrix, class_report


