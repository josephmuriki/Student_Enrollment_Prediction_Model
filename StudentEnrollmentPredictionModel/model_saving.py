# model_saving.py
import joblib

def model_saving(model, filename='student_enrollment_prediction_model.pkl'):

    # Saving the trained model to a file
    joblib.dump(model, filename)
    print(f"Model saved to {filename}.")
