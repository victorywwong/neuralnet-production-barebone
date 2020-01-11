import sys
import os
import random

from tqdm import tqdm

from flask import Blueprint, request, jsonify, Flask
from flask_cors import CORS
import torch
import torch.nn.functional as F

import db
import config
from ml.model import Net
from ml.utils import prediction_result

app = Flask(__name__)
api = Blueprint('api', __name__)

# Making a Cross domain request
CORS(api)

# Load pytorch model for inference
model_name = 'model_en.pth'
model_path = f'./ml/models/{model_name}'
model = Net()

if torch.cuda.is_available():
    trained_weights = torch.load(model_path)
else:
    trained_weights = torch.load(model_path, map_location='cpu')

model.load_state_dict(trained_weights)
model.eval()
print('PyTorch model loaded !')

@app.route('/healthcheck', methods=['GET', 'POST'])
def healthy():
    return jsonify(success=True)


@api.route('/predict', methods=['POST'])
def predict_rating():
    '''
    Endpoint to predict the rating using the
    experiments's text data.
    '''
    if request.method == 'POST':
        if 'experiment' not in request.form:
            return jsonify({'error': 'no experiment in body'}), 400
        else:
            parameters = model.get_model_parameters()
            experiment = request.form['experiment']
            output = prediction_result(model, experiment, **parameters)
            return jsonify(float(output))


@api.route('/submit', methods=['POST'])
def post_experiment():
    '''
    Save experiment to database.
    '''
    if request.method == 'POST':
        expected_fields = [
            'variable_a',
            'variable_b'
            'user_agent',
            'ip_address'
        ]
        if any(field not in request.form for field in expected_fields):
            return jsonify({'error': 'Missing field in body'}), 400

        query = db.Experiment.create(**request.form)

        return jsonify(query.serialize())


app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=config.DEBUG, host=config.HOST)
