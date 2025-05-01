from base_iris_lab1 import models, datasets
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from datetime import datetime
import json
from post_score import post_score
from lab4_header import scores_table
from lambda_test import lambda_handler

 
metrics = []

def save_metrics( model_id, metrics_bundle ):
    if model_id < len(metrics):
        metrics[model_id] = metrics_bundle
    else:
        metrics.append(metrics_bundle)
    return

def test( model_id, dataset_id ):
    global models, datasets, metrics

    model = models[model_id]
    df = datasets[dataset_id]

    X_test = df.iloc[:,1:21].values
    y = df.iloc[:,0].values

    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    y_test = pd.get_dummies(y1).values

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(X_test)

    actual = np.argmax(y_test,axis=1)
    predicted = np.argmax(y_pred,axis=1)
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")

    # Write each prediction to IrisExtended using post_score
    for i in range(len(X_test)):
        feature_string = ','.join(map(str, X_test[i]))
        class_string = str(predicted[i])
        actual_string = str(actual[i])
        # Use actual prediction probabilities, or simulate low confidence for testing
        prob = y_pred[i]
        # Simulate low confidence for ~20% of records (for testing)
        import random
        if random.random() < 0.2:
            prob = [0.8, 0.1, 0.1]  # Simulated low-confidence probabilities
        prob_string = ','.join(map(str, prob))
        
        # Generate partition_key as timestamp
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Write to IrisExtended
        post_score(
            log_table=scores_table,
            feature_string=feature_string,
            class_string=class_string,
            actual_string=actual_string,
            prob_string=prob_string
        )


    save_metrics(model_id, {'accuracy' : accuracy, 'actual': actual.tolist(), 'predicted':predicted.tolist()})

    return( metrics[model_id] )
