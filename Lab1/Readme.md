# Lab 1: REST API for Extended Iris Model

## Overview

This repository contains the implementation of a REST API for an extended Iris classification model. The project was developed as part of the EAI Systems Spring 2025 Mini 2, Assignment 1. The API is built using Flask and allows users to train and evaluate a deep learning model on an extended Iris dataset.

## Repository Structure
Lab1/
│-- codebase/
│   │-- base_iris.py         # Main model implementation
│   │-- hello_flask.py       # Basic Flask example
│
│-- screenshot/
│   │-- <Postman screenshots verifying API functionality>
│
│-- logs/
│   │-- <Training logs for model training>
│
│-- ewepngon_summary.pdf     # Summary of assigned reading articles
│-- README.md                # This file

## Installation and Setup

### Prerequisites

Ensure you have Python and the required dependencies installed:

pip install numpy pandas tensorflow sklearn flask

Navigate to the Lab1/codebase directory:

cd Lab1/codebase

Run the Flask API:

python hello_flask.py

The API should now be running on http://localhost:4000

### API Endpoints

1. Train the Model

Endpoint: POST /train

Description: Trains a new model using the extended Iris dataset.

Response:

{
    "message": "Model trained successfully!",
    "model_ID": <model_id>,
    "training_history": {...}
}

2. Make a Prediction

Endpoint: POST /predict

Request Body:

{
    "features": [f1, f2, ..., f20]
}

Response:

{
    "prediction": "Predicted class: <class_label>"
}

Screenshots

Screenshots of Postman tests for the API endpoints are available in the screenshot/ folder.

Logs

Training logs for the model can be found in the logs/ folder.

Summary of Readings

The document ewepngon_summary.pdf contains a summary of the assigned readings on HTTP and Web resources.