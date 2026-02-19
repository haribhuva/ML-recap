import joblib
import os

def save_encoders(le_location, le_item, path='model/'):
    """
    Save label encoders to files.
    """
    os.makedirs(path, exist_ok=True)
    joblib.dump(le_location, os.path.join(path, 'le_location.pkl'))
    joblib.dump(le_item, os.path.join(path, 'le_item.pkl'))

def load_encoders(path='model/'):
    """
    Load label encoders from files.
    """
    le_location = joblib.load(os.path.join(path, 'le_location.pkl'))
    le_item = joblib.load(os.path.join(path, 'le_item.pkl'))
    return le_location, le_item

def encode_test_data(test_df, le_location, le_item):
    """
    Encode categorical variables in test data using saved encoders.
    """
    test_df['locationId_encoded'] = le_location.transform(test_df['locationId'])
    test_df['item_id_encoded'] = le_item.transform(test_df['item_id'])
    return test_df