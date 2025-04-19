# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
from dash import Dash, dcc, html, Input, Output, State
from dash import Dash, dash_table

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

col_style = {'display':'grid', 'grid-auto-flow': 'row'}
row_style = {'display':'grid', 'grid-auto-flow': 'column'}

import plotly.express as px
import pandas as pd
import requests
import json
import io

# Base URL for API requests
BASE_URL = "http://localhost:4000/iris"

app = Dash(__name__)

# Load initial data
df = pd.read_csv("iris_extended_encoded.csv", sep=',')
df_csv = df.to_csv(index=False)

app.layout = html.Div(children=[
    html.H1(children='Iris classifier'),
    dcc.Tabs([
    dcc.Tab(label="Explore Iris training data", style=tab_style, selected_style=tab_selected_style, children=[

    html.Div([
        html.Div([
            html.Label(['File name to Load for training or testing'], style={'font-weight': 'bold'}),
            dcc.Input(id='file-for-train', type='text', style={'width':'100px'}),
            html.Div([
                html.Button('Load', id='load-val', style={"width":"60px", "height":"30px"}),
                html.Div(id='load-response', children='Click to load')
            ], style=col_style)
        ], style=col_style),

        html.Div([
            html.Button('Upload', id='upload-val', style={"width":"60px", "height":"30px"}),
            html.Div(id='upload-response', children='Click to upload')
        ], style=col_style| {'margin-top':'20px'})

    ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),


html.Div([
    html.Div([
        html.Div([
            html.Label(['Feature'], style={'font-weight': 'bold'}),
            dcc.Dropdown(
                df.columns[1:],  # All columns except species
                df.columns[1],   # Default to first feature
                id='hist-column'
            )
            ], style=col_style ),
        dcc.Graph( id='selected_hist' )
    ], style=col_style | {'height':'400px', 'width':'400px'}),

    html.Div([

    html.Div([

    html.Div([
        html.Label(['X-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            df.columns[1:],  # All columns except species
            df.columns[3],   # Default to sepal_length
            id='xaxis-column'
            )
        ]),

    html.Div([
        html.Label(['Y-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            df.columns[1:],  # All columns except species
            df.columns[4],   # Default to sepal_width
            id='yaxis-column'
            )
        ])
    ], style=row_style | {'margin-left':'50px', 'margin-right': '50px'}),

    dcc.Graph(id='indicator-graphic')
    ], style=col_style)
], style=row_style),

    html.Div(id='tablecontainer', children=[
        dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=15,
            id='datatable' )
        ])
    ]),
    dcc.Tab(label="Build model and perform training", id="train-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Button('New model', id='build-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='build-response', children='Click to build new model and train')
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Label(['Enter a model ID for re-training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Re-Train', id='train-val', style={"width":"90px", "height":"30px"})
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-train', children='')
    ]),
    dcc.Tab(label="Score model", id="score-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a row text (CSV) to use in scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='row-for-score', type='text', style={'width':'300px'}))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID for scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-score', type='text'))
            ], style=col_style | {'margin-top':'20px'}),            
            html.Div([
                html.Button('Score', id='score-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='score-response', children='Click to score')
            ], style=col_style | {'margin-top':'20px'})
        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),
        
        html.Div(id='container-button-score', children='')
    ]),

    dcc.Tab(label="Test Iris data", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Test', id='test-val'),
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-test', children='')
    ])

    ])
])

# callbacks for Explore data tab

@app.callback(
    Output('load-response', 'children'),
    Input('load-val', 'n_clicks'),
    State('file-for-train', 'value')
)
def update_output_load(nclicks, filename):
    global df, df_csv

    if nclicks is not None and filename:
        try:
            # load local data given input filename
            df = pd.read_csv(filename, sep=',')
            df_csv = df.to_csv(index=False)
            return f'Load done. Loaded {len(df)} records from {filename}.'
        except Exception as e:
            return f'Error loading file: {str(e)}'
    else:
        return 'Click to load'


@app.callback(
    Output('build-response', 'children'),
    Input('build-val', 'n_clicks'),
    State('dataset-for-train', 'value')
)
def update_output_build(nclicks, dataset_id):
    if nclicks is not None and dataset_id:
        try:
            # invoke new model endpoint to build and train model given data set ID
            response = requests.post(f"{BASE_URL}/model", data={"dataset": dataset_id})
            if response.status_code == 200:
                result = response.json()
                # return the model ID 
                return f'Model built and trained with ID: {result["model_ID"]}'
            else:
                return f'Error: {response.text}'
        except Exception as e:
            return f'Error: {str(e)}'
    else:
        return 'Click to build new model and train'

@app.callback(
    Output('upload-response', 'children'),
    Input('upload-val', 'n_clicks')
)
def update_output_upload(nclicks):
    global df_csv

    if nclicks is not None:
        try:
            # invoke the upload API endpoint
            response = requests.post(f"{BASE_URL}/datasets", data={"train": df_csv})
            if response.status_code == 200:
                result = response.json()
                # return the dataset ID generated
                return f'Dataset uploaded with ID: {result["dataset_ID"]}'
            else:
                return f'Error: {response.text}'
        except Exception as e:
            return f'Error: {str(e)}'
    else:
        return 'Click to upload'

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('load-val', 'n_clicks')]
)
def update_graph(xaxis_column_name, yaxis_column_name, n_clicks):
    if xaxis_column_name is None or yaxis_column_name is None:
        # Provide default values if none are selected
        xaxis_column_name = df.columns[3]  # sepal_length
        yaxis_column_name = df.columns[4]  # sepal_width

    fig = px.scatter(df, 
                    x=xaxis_column_name,
                    y=yaxis_column_name,
                    color='species',
                    title=f'{yaxis_column_name} vs {xaxis_column_name}')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 50, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=xaxis_column_name)
    fig.update_yaxes(title=yaxis_column_name)

    return fig


@app.callback(
    Output('selected_hist', 'figure'),
    [Input('hist-column', 'value'),
     Input('load-val', 'n_clicks')]
)
def update_hist(hist_column_name, n_clicks):
    if hist_column_name is None:
        # Provide default value if none is selected
        hist_column_name = df.columns[1]

    fig = px.histogram(df, x=hist_column_name, color='species',
                      title=f'Distribution of {hist_column_name}')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 50, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=hist_column_name)

    return fig

@app.callback(
    Output('datatable', 'data'),
    Output('datatable', 'columns'),
    [Input('load-val', 'n_clicks')]
)
def update_table(n_clicks):
    columns = [{"name": i, "id": i} for i in df.columns]
    return df.to_dict('records'), columns


# Store training history between runs
training_histories = {}

# callbacks for Training tab
@app.callback(
    Output('container-button-train', 'children'),
    Input('train-val', 'n_clicks'),
    State('model-for-train', 'value'),
    State('dataset-for-train', 'value')
)
def update_output_train(nclicks, model_id, dataset_id):
    global training_histories
    
    if nclicks is not None and model_id and dataset_id:
        try:
            # Retrain model API endpoint request
            r = requests.put(f"{BASE_URL}/model/{model_id}", params={"dataset": dataset_id})
            
            if r.status_code == 200:
                # Extract history from response
                data = r.json()
                current_history = data.get('history', {})
                
                # Initialize history for this model if it doesn't exist
                if model_id not in training_histories:
                    training_histories[model_id] = []
                
                # Add current training run to history
                run_number = len(training_histories[model_id]) + 1
                
                # Convert history to dataframe format
                train_data = []
                for epoch, acc in enumerate(current_history.get('accuracy', [])):
                    train_data.append({
                        'epoch': epoch, 
                        'accuracy': acc,
                        'loss': current_history.get('loss', [])[epoch] if epoch < len(current_history.get('loss', [])) else None,
                        'run': f'Run {run_number}'
                    })
                
                # Save this training run
                training_histories[model_id].append(train_data)
                
                # Combine all training runs for this model
                all_runs = []
                for i, hist in enumerate(training_histories[model_id]):
                    all_runs.extend(hist)
                
                combined_df = pd.DataFrame(all_runs)
                
                # Create figures showing all training runs
                acc_fig = px.line(combined_df, x='epoch', y='accuracy', color='run', 
                                 title='Training Accuracy History')
                acc_fig.update_layout(xaxis_title='Epoch', yaxis_title='Accuracy')
                
                loss_fig = px.line(combined_df, x='epoch', y='loss', color='run', 
                                  title='Training Loss History')
                loss_fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss')
                
                return html.Div([
                    html.H4('Training Results History'),
                    html.P(f"Model ID: {model_id}, Training Run: {run_number}"),
                    dcc.Graph(figure=acc_fig),
                    dcc.Graph(figure=loss_fig)
                ])
            else:
                return html.Div(f"Error: {r.text}")
        except Exception as e:
            return html.Div(f"Error: {str(e)}")
    else:
        return ""

# Define a mapping of class numbers to species names
IRIS_CLASSES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# callbacks for Scoring tab
@app.callback(
    Output('container-button-score', 'children'),
    Input('score-val', 'n_clicks'),
    State('model-for-score', 'value'),
    State('row-for-score', 'value')
)
def update_output_score(nclicks, model_id, row_text):
    if nclicks is not None and model_id and row_text:
        try:
            # Parse comma-separated values
            feature_values = [float(x.strip()) for x in row_text.split(',')]
            
            # Check if we have the expected number of features (20)
            if len(feature_values) != 20:
                return f"Error: Expected 20 feature values, got {len(feature_values)}"
            
            # Add API endpoint request for scoring here with constructed input row
            fields_param = ','.join(map(str, feature_values))
            r = requests.get(f"{BASE_URL}/model/{model_id}/score", params={"fields": fields_param})
            
            if r.status_code == 200:
                score_result = r.json()
                
                # Extract result text and parse class number
                result_text = score_result.get('result', 'No result returned')
                
                # Parse the class number from the result string
                # Assuming format like "Score done, class=0"
                try:
                    class_num = int(result_text.split('=')[-1].strip())
                    species_name = IRIS_CLASSES.get(class_num, f"Unknown (Class {class_num})")
                    
                    # Format the result with both class number and name
                    formatted_result = f"Class {class_num} - {species_name}"
                except (ValueError, IndexError):
                    formatted_result = result_text
                
                # Create a more detailed and formatted display
                return html.Div([
                    html.H4('Scoring Results'),
                    html.Div([
                        html.P(f"Model ID: {model_id}"),
                        html.P([
                            "Raw Result: ", 
                            html.Span(result_text, style={'fontFamily': 'monospace'})
                        ]),
                        html.P([
                            "Interpreted Result: ", 
                            html.Span(formatted_result, style={'fontWeight': 'bold', 'color': '#007bff'})
                        ]),
                        html.Hr(),
                        html.Details([
                            html.Summary("Input Features"),
                            html.Pre(", ".join([f"{v:.2f}" for v in feature_values]), 
                                    style={'backgroundColor': '#f8f9fa', 'padding': '10px'})
                        ])
                    ], style={'border': '1px solid #ddd', 'padding': '15px', 'marginTop': '10px', 
                              'borderRadius': '5px', 'backgroundColor': '#f8f9fa'})
                ])
            else:
                return f"Error: {r.text}"
        except Exception as e:
            return f"Error: {str(e)}"
    else:
        return ""
    

# callbacks for Testing tab

@app.callback(
    Output('container-button-test', 'children'),
    Input('test-val', 'n_clicks'),
    State('model-for-test', 'value'),
    State('dataset-for-test', 'value')
)
def update_output_test(nclicks, model_id, dataset_id):
    if nclicks is not None and model_id and dataset_id:
        try:
            # Add API endpoint request for testing with given dataset ID
            r = requests.get(f"{BASE_URL}/model/{model_id}/test", params={"dataset": dataset_id})
            
            if r.status_code == 200:
                test_results = r.json()
                
                # Extract metrics
                metrics = test_results.get('metrics', {})
                accuracy = metrics.get('accuracy', 0)
                
                # Extract actual and predicted values if available
                actual = metrics.get('actual', [])
                predicted = metrics.get('predicted', [])
                
                # Create confusion matrix data if we have predictions
                if actual and predicted:
                    # Count occurrences of each combination
                    confusion_data = []
                    for i in range(len(actual)):
                        confusion_data.append({
                            'actual': f"Class {actual[i]}",
                            'predicted': f"Class {predicted[i]}"
                        })
                    
                    confusion_df = pd.DataFrame(confusion_data)
                    
                    # Create heatmap of confusion matrix
                    confusion_fig = px.density_heatmap(
                        confusion_df, 
                        x="predicted", 
                        y="actual",
                        title="Confusion Matrix",
                        labels={"predicted": "Predicted Class", "actual": "Actual Class"}
                    )
                    
                    return html.Div([
                        html.H4('Test Results'),
                        html.Div([
                            html.P(f"Accuracy: {accuracy:.4f}"),
                        ], style={'border': '1px solid #ddd', 'padding': '10px', 'marginTop': '10px'}),
                        dcc.Graph(figure=confusion_fig)
                    ])
                else:
                    # Just display accuracy if we don't have prediction details
                    return html.Div([
                        html.H4('Test Results'),
                        html.Div([
                            html.P(f"Accuracy: {accuracy:.4f}"),
                        ], style={'border': '1px solid #ddd', 'padding': '10px', 'marginTop': '10px'})
                    ])
            else:
                return f"Error: {r.text}"
        except Exception as e:
            return f"Error: {str(e)}"
    else:
        return ""


if __name__ == '__main__':
    app.run(debug=True)