from routes.data_load import load_data
from routes.preprocessing import preprocess_data, split_data
from routes.model import train_model, evaluate_model
from routes.encoding import save_encoders

def main():
    print("Starting ML Pipeline...")

    # Load data
    df = load_data()
    print(f"Loaded data with shape: {df.shape}")

    # Preprocess
    X, y, le_location, le_item = preprocess_data(df)
    print("Data preprocessed.")

    # Save encoders
    save_encoders(le_location, le_item)
    print("Encoders saved.")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Data split into train and test.")

    # Train model
    model = train_model(X_train, y_train)
    print("Model trained and logged to MLflow.")

    # Evaluate
    evaluate_model(model, X_test, y_test)
    print("Model evaluated.")

    print("ML Pipeline completed!")

if __name__ == "__main__":
    main()
