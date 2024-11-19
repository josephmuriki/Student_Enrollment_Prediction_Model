# predictions_export.py
import pandas as pd

def export_predictions(X_test, y_test, y_pred, filename='student_enrollment_predictions.csv'):

    # Adding predictions
    results = X_test.copy()
    results['Actual'] = y_test
    results['Predicted'] = y_pred

    # Saving the predictions to a CSV file
    results.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}.")
