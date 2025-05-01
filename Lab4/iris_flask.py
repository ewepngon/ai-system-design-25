from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from base_iris_lab1 import add_dataset, new_model, train, score
import io
from test import test

app = Flask(__name__)

# Testing the server communication
@app.route('/')
def home():
    return "Hello, World!"

# Upload the training data
@app.route('/iris/datasets', methods=['POST'])
def upload_dataset():
    if 'train' not in request.form:
        return jsonify({"error": "No dataset  provided"}), 400
    

    file = request.form['train']
    if not file:
        return jsonify({"error": "No file provided"}), 400
    try:
        # split line by line and extract header and then create the dataframe
        file = file.split('\n')
        header = file[0]
        file = file[1:]
        file = '\n'.join(file)
        file = io.StringIO(file)
    except Exception as e:
        return jsonify({"error": "Invalid file format"}), 400
    
    df = pd.read_csv(file)
    dataset_ID = add_dataset(df)
    
    return jsonify({"dataset_ID": dataset_ID})


# Build and train the new model instance

@app.route('/iris/model', methods=['POST'])
def create_model():
    if 'dataset' not in request.form:
        return jsonify({"error": "No dataset index provided"}), 400
    
    print("We are here")

    dataset_ID = request.form['dataset']
    try:
        dataset_ID = int(dataset_ID)
       
    except Exception as e:
        return jsonify({"error": "Invalid dataset index"}), 400
    
    model_ID = new_model(dataset_ID)
    return jsonify({"model_ID": model_ID})
    


# Re-train the model using specified dataset
@app.route('/iris/model/<int:model_ID>', methods=['PUT'])
def retrain_model(model_ID):
    dataset_ID = request.args.get('dataset')
    if dataset_ID is None:
        return jsonify({"error": "Dataset index required"}), 400
    try:
        dataset_ID = int(dataset_ID)
    except Exception as e:
        return jsonify({"error": "Invalid dataset index"}), 400
    
    history = train(model_ID, dataset_ID)
    return jsonify({"history": history})

# Score the model with provided values
@app.route('/iris/model/<int:model_ID>/score', methods=['GET'])
def score_model(model_ID):
    fields = request.args.get('fields')
    if fields is None:
        return jsonify({"error": "No fields provided"}), 400
    
    try:
        features = list(map(float, fields.split(',')))
    except Exception as e:
        return jsonify({"error": "Invalid feature values"}), 400
    
    
    if len(features) != 20:
        return jsonify({"error": "Exactly 20 feature values required"}), 400
    
    result = score(model_ID, features)
    return jsonify({"result": result})


@app.route('/iris/model/<int:model_ID>/test', methods=['GET'])
def test_model(model_ID):
    dataset_ID = request.args.get('dataset')

    if dataset_ID is None:
        return jsonify({"error": "Dataset index required"}), 400
    try:
        dataset_ID = int(dataset_ID)
    except ValueError:
        return jsonify({"error": "Invalid dataset index"}), 400

    try:
        test_results = test(model_ID, dataset_ID)  # Call test function
    except Exception as e:
        return jsonify({"error": f"Failed to test model: {str(e)}"}), 500

    return jsonify({"metrics": test_results})  # Return test results as JSON

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
