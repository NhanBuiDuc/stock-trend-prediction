import os
from flask import Flask, jsonify, render_template, request, json
from flask_cors import CORS, cross_origin
import subprocess
import pandas as pd
import base64
import shutil
from pathlib import Path
import requests
from APP_FLASK.TrendPrediction import Predictor

app = Flask(__name__)
predictor = Predictor()
CORS(app)
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

curr_path = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = ROOT_DIR.replace('\\', '/')
os.chdir(ROOT_DIR.split("APP_WEB")[0])
print(os.getcwd())
CODE_DIR = ROOT_DIR + '/APPWEB'


@app.route('/execute/<file>', methods=['GET'])
def predict_LSTM(file):
    symbol = file
    print('asdasd: '+ ROOT_DIR.split("APP_WEB")[0])
    try:
        # Execute your Python file using subprocess module
        # subprocess.run(['python', './predict_stock.py'], check=True)
        # returnValue = subprocess.check_output(
        # ['python', './predict_stock.py'])

        createFile(symbol)
        returnValue = subprocess.check_output(
                "python predict_stock.py")

        json_str = json.dumps(
            {'price': returnValue.decode('utf-8').replace("\\", "").replace("\r", "").replace("\n", "").split("tensor")[1].split("(")[1].split(")")[0]})
        formated = json.loads(json_str)
        # return jsonify(json_str)
        return formated
    except Exception as e:
        return f'Error executing Python file: {str(e)}'


@app.route('/', methods=['GET'])
def index1():
    return render_template('index.html')


def createFile(content):
    #symbolconfig.json {'symbol': 'GOOGLE'}
    FILE = 'symbolconfig.json'
    path = './configs/' + FILE
    
    # Create and write data to text file
    with open(path, 'w') as fp:
        fp.write('{"symbol":"' + content + '"}')

@cross_origin(supports_credentials=True)
@app.route("/trend_prediction", methods = ['GET', 'POST'])
def Predict_trend():
    try:
        args = 	request.args
        symbol = args.get('symbol')
        window_size = args.get('window_size')
        output_size = args.get('output_size')
        model_type_list = args.get('output_size')
        output_dict = predictor.batch_predict(symbol, model_type_list, window_size, output_size)
        response = app.response_class(
            response=json.dumps(output_dict),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        return f'Error executing Python file: {str(e)}'
	



if __name__ == '__main__':
    app.run(debug=True)


