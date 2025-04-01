import requests
import json

def train_model():
    response = requests.post("http://localhost:4000/train")
    print("Train response: ", response.json())
    return response.json()

def predict(features):
    payload = {"features": features}
    response = requests.post("http://localhost:4000/predict", json=payload)
    print("Predict response: ", response.json())
    return response.json()

def test_model(model_id, dataset_id):
    url = f"http://localhost:4000/iris/model/{model_id}?dataset={dataset_id}"
    response = requests.get(url)
    # print("Test response: ", response.json())
    return response.json()

def main():
    print("Starting client...")
    
    # Train model
    print("Training model...")
    train_response = train_model()
    # print("Response:", train_response)
    
    if "model_ID" not in train_response:
        print("Error: Model training failed.")
        return
    
    model_id = train_response["model_ID"]
    
    # Test prediction
    sample_features = [0.5] * 20  # Replace with actual test data
    print("Making a prediction...")
    prediction_response = predict(sample_features)
    
    # Test model
    dataset_id = 0  # Replace with the correct dataset ID
    print(f"Testing model {model_id} with dataset {dataset_id}...")
    test_response = test_model(model_id, dataset_id)

if __name__ == "__main__":
    main()