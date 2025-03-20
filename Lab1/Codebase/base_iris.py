import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

print('Starting up the extended Iris model service')

# Global storage for models and datasets
global models, datasets
models = []
datasets = []

def build():
    """Builds a model with 20 input features"""
    global models

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20,)),  # ✅ Ensure correct input size
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 output classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    models.append(model)
    return len(models) - 1

def load_local():
    """Loads the extended dataset and ensures it has 20 features + 1 target column"""
    global datasets
    print("Loading extended Iris dataset...")

    dataFile = "./iris_extended_encoded.csv"
    dataset = pd.read_csv(dataFile)

    # ✅ Debugging Step: Check Columns
    print(f"Dataset Columns: {list(dataset.columns)}")
    print(f"Dataset Shape: {dataset.shape}")

    # ✅ Ensure the dataset has 21 columns (20 features + 1 label)
    if dataset.shape[1] != 21:
        raise ValueError(f"Expected dataset with 21 columns, but got {dataset.shape[1]}")

    datasets.append(dataset)
    return len(datasets) - 1

def train(model_ID, dataset_ID):
    global datasets, models
    dataset = datasets[dataset_ID]
    model = models[model_ID]

    # ✅ Select 20 feature columns
    X = dataset.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values
    y = dataset.iloc[:, 0].values  # ✅ Select target column

    # ✅ Handle missing values
    X = np.nan_to_num(X)

    # ✅ Encode categorical labels
    encoder = LabelEncoder()
    y1 = encoder.fit_transform(y)
    Y = pd.get_dummies(y1).values

    # ✅ Check if dataset shape is correct
    print(f"Feature Shape: {X.shape}, Target Shape: {Y.shape}")

    # ✅ Ensure X has 20 features
    if X.shape[1] != 20:
        raise ValueError(f"Expected 20 features, but got {X.shape[1]}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Train the model
    history = model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)
    print(history.history)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    return history.history

def new_model(dataset_ID):
    """Creates and trains a model in one function"""
    model_ID = build()
    history = train(model_ID, dataset_ID)
    return model_ID, history

def score(model_ID, *features):
    """Scores a new sample with 20 feature values"""
    global models
    model = models[model_ID]

    # ✅ Ensure exactly 20 input features
    if len(features) != 20:
        raise ValueError(f"Expected 20 features, but got {len(features)}")

    x_test = np.array([features])  # Convert list to numpy array

    y_pred = model.predict(x_test)
    iris_class = np.argmax(y_pred, axis=1)[0]
    return f"Predicted class: {iris_class}"
