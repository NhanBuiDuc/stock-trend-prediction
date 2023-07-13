# point to parent folder
import os
import sys

curr_path = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = ROOT_DIR.replace('\\', '/')
#os.chdir(ROOT_DIR.split("APP_WEB")[0])
sys.path.append(ROOT_DIR.split("APP_WEB")[0])

#import libraries
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

CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/execute/<file>', methods=['GET'])
def predict_LSTM(file):
    symbol = file
    

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

# @app.route('/chart', methods=['GET','POST'])
# def my_route():
#     symbol = request.args.get('symbol', default = '*', type = str)
#     windowsize = request.args.get('windowsize', default = 7, type = int) 
#     outputsize = request.args.get('outputsize', default = 7, type = int)


#     print(symbol,windowsize,outputsize) 
#     response = app.response_class(
# 		response=json.dumps({'symbol': symbol, 'windowsize': windowsize, 'outputsize': outputsize}),
# 		status=200,
# 		mimetype='application/json'
# 	)
#     return response

'''@cross_origin(supports_credentials=True)
@app.route("/execute", methods = ['POST'])
def Predict_trend():
    data = json.loads(request.data)
    print(data)
    
    # try: 
    # args = 	request.args
    # symbol = args.get('symbol', default='AAPL', type=str)
    # window_size = args.get('window_size', default = 7, type = int)
    # output_size = args.get('output_size', default = 7, type = int)
    # model_type_list = args.get('model_type_list', type=list)
    # print('model tpe', args.get('model_type_list'))

    output_dict = predictor.batch_predict(data['symbol'], data['model_type_list'], data['window_size'], data['output_size'])
    print('kkkkkkkkkkkkkkkkkkk', output_dict)
    response = app.response_class(
        response=json.dumps(output_dict),
        status=200,
        mimetype='application/json'
    )
    return response
    # except Exception as e:
    #     return f'Error executing Python file: {str(e)}'
'''

if __name__ == '__main__':
    app.run(debug=True)

