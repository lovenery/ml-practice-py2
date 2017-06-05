import numpy as np
from flask import Flask, abort, jsonify, request
import cPickle as pickle
import os.path as path

output_file = path.join(path.dirname(__file__), './iris_rfc.pkl')
my_random_forest = pickle.load(open(output_file, "rb"))

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def make_predict():

    #TODO: error cheching

    # convert to np array
    data = request.get_json(force=True)
    predict_request = [ data['sl'], data['sw'], data['pl'], data['pw'] ]
    predict_request = np.array(predict_request)

    # prediction
    y_hat = my_random_forest.predict(predict_request)
    output = [y_hat[0]]
    return jsonify(results = output)

if __name__ == '__main__':
    app.run(port = 5000, debug = True)