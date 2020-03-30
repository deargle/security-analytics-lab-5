import pickle as pkl
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

with open(os.getenv('MODEL_NAME'), 'rb') as f:
    model = pkl.load(f)

model_n_features = model.named_steps['preprocessor']._n_features

'''
the model pipeline was fit on a dataset that had all of the following
columns. It needs to be given all of the columns even if it ends
up dropping most of them.

`model_features` says what the pipeline eventually retains.
'''
original_dataset_features = ['duration',
 'protocol_type',
 'service',
 'flag',
 'src_bytes',
 'dst_bytes',
 'land',
 'wrong_fragment',
 'urgent',
 'hot',
 'num_failed_logins',
 'logged_in',
 'lnum_compromised',
 'lroot_shell',
 'lsu_attempted',
 'lnum_root',
 'lnum_file_creations',
 'lnum_shells',
 'lnum_access_files',
 'lnum_outbound_cmds',
 'is_host_login',
 'is_guest_login',
 'count',
 'srv_count',
 'serror_rate',
 'srv_serror_rate',
 'rerror_rate',
 'srv_rerror_rate',
 'same_srv_rate',
 'diff_srv_rate',
 'srv_diff_host_rate',
 'dst_host_count',
 'dst_host_srv_count',
 'dst_host_same_srv_rate',
 'dst_host_diff_srv_rate',
 'dst_host_same_src_port_rate',
 'dst_host_srv_diff_host_rate',
 'dst_host_serror_rate',
 'dst_host_srv_serror_rate',
 'dst_host_rerror_rate',
 'dst_host_srv_rerror_rate']

feature_names = [str(name) for name in os.getenv('MODEL_FEATURES').split(',')]


@app.route('/')
def hello_world():
    return 'Hello, World!'

#https://flask.palletsprojects.com/en/1.1.x/patterns/apierrors/
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

def make_prediction(data):
    predict_me = {feature_name: None for feature_name in original_dataset_features}
    for feature_name in feature_names:
        submitted_val = data[feature_name]
        if not submitted_val:
            raise InvalidUsage('missing feature {}'.format(feature_name), 400)
        predict_me[feature_name] = submitted_val

    predict_me = pd.DataFrame(predict_me, index=[0]) # it's not typical to build a
                                                     # dataframe with only one row, but that's
                                                     # what we're doing, so pandas wants us to
                                                     # specify the index for that row with `index=[0]`
    y_pred = '{:.3f}'.format(model.predict_proba(predict_me)[:,1][0])
    return y_pred

#https://flask.palletsprojects.com/en/1.1.x/quickstart/#the-request-object
@app.route('/predict.json', methods=['POST'])
def predict_json():
    data = request.get_json(force=True)
    # override the features we actually care about with ones submitted by the form.
    try:
        y_pred = make_prediction(data)
    except InvalidUsage as e:
        response = jsonify(e.to_dict())
        response.status_code = e.status_code
        return response
    return jsonify({'pred':y_pred})


@app.route('/predict', methods=['GET','POST'])
def predict():
    error = None
    y_pred = None
    if request.method == 'POST':
        data = request.form
        try:
            y_pred = make_prediction(data)
        except InvalidUsage as e:
            error = e.message # so that the render_template call below can access
    return render_template('predict.html', error=error, y_pred=y_pred, feature_names=feature_names, predictors=request.form)
