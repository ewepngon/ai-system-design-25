import json
import logging
import boto3

# Get the service resource.

client = boto3.client('dynamodb')

def lambda_handler(event, context):
    # TODO implement

    print(event)
    for rec in event['Records']:
        print(rec)
        if rec['eventName'] == 'INSERT':
            UpdateItem = rec['dynamodb']['NewImage']
            print(UpdateItem)

            # lab4 code goes here
            # Extract Class, Actual and Probability from the event
            predicted_class = UpdateItem['Class']['S']
            actual_class = UpdateItem['Actual']['S']
            probability = float(UpdateItem['Probability']['S'])

            # Check if the record is problematic
            if predicted_class != actual_class or probability < 0.9:
                print("Copying to the retrain table")
                
                response = client.put_item( TableName =  'IrisExtendedRetrain', Item = UpdateItem )
                print (response)

            else:
                print("Record is not problematic, no action taken")

    return {
        'statusCode': 200,
        'body': json.dumps( 'IrisExtendedRetrain Lambda return' )
    }
    
