# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
from service.recommend_service import RecommendService
import os
import pickle
from io import StringIO
import json
import flask

import pandas as pd
from service.predict_input_service import PredictInputService
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = RecommendService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    predict_input_service = PredictInputService()
    # Convert from CSV to pandas
    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        user_item_rating_df, item_tag_df = PredictInputService.get_input_data(
            PredictInputService.REQUEST_TYPE_ENDPOINT
            PredictInputService.REQUEST_CONTENT_TYPE_JSON
        )
        s = StringIO(data)
        json_data = json.load(s)
        top_n = json_data['top_n']
        user_item_rating_json = json_data['user_item_rating']
        item_tag_distr_json = json_data['item_tag']
        user_item_rating_json_str = json.dumps(user_item_rating_json)
        item_tag_distr_json_str = json.dumps(item_tag_distr_json)
        user_item_rating_df = pd.read_json(
            user_item_rating_json_str,
            orient='records'
        )
        item_tag_distr_df = pd.read_json(
            item_tag_distr_json_str,
            orient='records'
        )
        # drop tag column
    elif flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO(data)
        s3_url_df = pd.read_csv(s, header=None)  
        user_item_rating_df = 
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

    #print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    prediction_df = RecommendService.predict(
        user_item_rating_df,
        item_tag_distr_df,
        top_n
    )
    print('prediction_df: ')
    print(prediction_df)
    # Convert from numpy back to CSV
    out = StringIO()
    prediction_df.to_csv(out, header=True, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
