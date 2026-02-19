import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Preprocess the data: handle missing values, encode categorical variables.
    """
    # Handle missing values
    df['onpromotion'] = df['onpromotion'].fillna(False)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Extract features from date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek

    # Encode categorical variables
    le_location = LabelEncoder()
    df['locationId_encoded'] = le_location.fit_transform(df['locationId'])

    le_item = LabelEncoder()
    df['item_id_encoded'] = le_item.fit_transform(df['item_id'])

    # Select features
    features = ['year', 'month', 'day', 'dayofweek', 'locationId_encoded', 'item_id_encoded', 'onpromotion']
    target = 'unit_sales'

    X = df[features]
    y = df[target]

    return X, y, le_location, le_item

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test