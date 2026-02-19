import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model locally
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.pkl')

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2: {r2}")

    return mae, mse, r2

def load_model(model_path='model/model.pkl'):
    """
    Load a trained model from file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model