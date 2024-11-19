# model_training.py
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):

    # Initializing the Random Forest classifier
    model = RandomForestClassifier(random_state=42)

    # Training the model
    model.fit(X_train, y_train)

    return model
