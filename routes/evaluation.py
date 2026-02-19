import pandas as pd
from routes.model import load_model
from routes.preprocessing import preprocess_data

def predict_on_test(test_df, model_path='model/model.pkl'):
    """
    Make predictions on test data.
    """
    model = load_model(model_path)

    # Preprocess test data (without target)
    test_df['onpromotion'] = test_df['onpromotion'].fillna(False)
    test_df['date'] = pd.to_datetime(test_df['date'])
    test_df['year'] = test_df['date'].dt.year
    test_df['month'] = test_df['date'].dt.month
    test_df['day'] = test_df['date'].dt.day
    test_df['dayofweek'] = test_df['date'].dt.dayofweek

    # Assuming encoders are saved or we need to handle encoding
    # For simplicity, assuming we have the encoders from training
    # In real MLOps, save encoders too

    features = ['year', 'month', 'day', 'dayofweek', 'locationId', 'item_id', 'onpromotion']
    # Note: Need to encode locationId and item_id similarly as in training

    # Placeholder: for now, assume encoded columns exist or handle encoding
    # In practice, load saved encoders

    X_test = test_df[features]  # This will fail if not encoded

    predictions = model.predict(X_test)
    return predictions

def save_predictions(predictions, output_path='predictions.csv'):
    """
    Save predictions to CSV.
    """
    pd.DataFrame(predictions, columns=['unit_sales']).to_csv(output_path, index=False)