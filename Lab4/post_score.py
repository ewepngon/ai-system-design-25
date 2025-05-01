from datetime import datetime

def post_score(log_table, feature_string, class_string, actual_string, prob_string ):
    current_time = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]

    response = log_table.put_item(
       Item={
            'partition_key': current_time,
            'sort_key': "abc",
            'Features': feature_string,
            'Class' : class_string,
            'Actual' : actual_string,
            'Probability' : prob_string
            }
    )
    return response
