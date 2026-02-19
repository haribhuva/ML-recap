from flask import Flask, request, jsonify
from routes.model import load_model
from routes.encoding import load_encoders, encode_test_data
import pandas as pd

app = Flask(__name__)

# Load model and encoders at startup
model = load_model()
le_location, le_item = load_encoders()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Expect data as list of dicts: [{'date': '2015-01-01', 'locationId': 'location_25', 'item_id': 'item_103665', 'onpromotion': False}, ...]

    df = pd.DataFrame(data)
    df = encode_test_data(df, le_location, le_item)

    # Preprocess
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek

    features = ['year', 'month', 'day', 'dayofweek', 'locationId_encoded', 'item_id_encoded', 'onpromotion']
    X = df[features]

    predictions = model.predict(X)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)