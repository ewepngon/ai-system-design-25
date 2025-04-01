from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from base_iris import load_local, build, train, score, new_model, train, test

app = Flask(__name__)

# Load the dataset
dataset_ID = load_local()  # Loads the extended dataset
model_ID = None  # Will be set when training happens

@app.route("/")
def index():
    return jsonify({"message": "Welcome to the Iris Classification API!"})

# ✅ Ensure model is trained before making predictions
@app.route("/train", methods=["POST"])
def train_model():
    global model_ID
    model_ID, history = new_model(dataset_ID)  # Build and train the model
    return jsonify({
        "message": "Model trained successfully!",
        "model_ID": model_ID,
        "training_history": history
    })

# ✅ Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    global model_ID

    try:
        # ✅ Ensure model is trained before predictions
        if model_ID is None:
            return jsonify({"error": "No trained model found. Train the model first using /train"}), 400

        # ✅ Get JSON input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON payload"}), 400

        # ✅ Validate "features" key
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        features = data["features"]

        # ✅ Ensure features is a list and has exactly 20 values
        if not isinstance(features, list):
            return jsonify({"error": "Expected a list of 20 features"}), 400

        if len(features) != 20:
            return jsonify({"error": f"Incorrect number of features. Expected 20, got {len(features)}"}), 400

        # ✅ Convert to NumPy array
        features = np.array(features).reshape(1, -1)

        # ✅ Make a prediction
        result = score(model_ID, *features.tolist()[0])
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/iris/model/<int:model_id>', methods=['GET'])
def test_model(model_id):
    dataset_ID = request.args.get("dataset")

    if dataset_ID is None:
        return jsonify({"error": "Missing dataset ID"}), 400
    
    try:
        results = test(model_id, int(dataset_ID))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4000)

