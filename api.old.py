import requests
import threading
import time
import face_recognition
import base64
import random
import os
import json
import numpy as np
from flask import Flask, request #import main Flask class and request object
app = Flask(__name__)
from flask import jsonify
from encoder import JsonEncoder
from wface import wface_encode;

@app.before_first_request
def activate_job():
    def run_job():
        while True:
            print("Run recurring task")
            time.sleep(3)

    thread = threading.Thread(target=run_job)
    thread.start()

def start_runner():
    def start_loop():
        not_started = True
        while not_started:
            print('In start loop')
            try:
                r = requests.get('http://127.0.0.1:5000/')
                if r.status_code == 200:
                    print('Server started, quiting start_loop')
                    not_started = False
                print(r.status_code)
            except:
                print('Server not yet started')
            time.sleep(2)

    print('Started runner')
    thread = threading.Thread(target=start_loop)
    thread.start()
	
@app.route("/", methods=['GET', 'POST'])
def hello():
	#language = request.args.get('language') #if key doesn't exist, returns None
	#language = request.form.get('test') #if key doesn't exist, returns None	
	print(request.json)
	return "Hello World!"

@app.route("/compare", methods=['GET', 'POST'])	
def compare():
	return "Com!"	

@app.route("/encode", methods=['GET', 'POST'])		
def encode():
	content = request.json
	image = content['image']
	#image = request.form.get('image') #if key doesn't exist, returns None
	format, imgstr = image.split(';base64,') 	
	ext = format.split('/')[-1] 	
	image = base64.b64decode(imgstr)	
	n = random.randint(0,100000);
	file_name = str(n) + "." + ext
	full_filename = os.path.join("tmp", file_name)
	with open(full_filename,"wb") as fo:
		fo.write(image)
		
	encode_value = wface_encode(full_filename)
	if encode_value is not None:
		dumps = json.dumps(encode_value,cls=JsonEncoder)
		os.remove(full_filename)
		return jsonify(
			encode=dumps,
			status="success"
		)	
	else:
		return jsonify(
			encode="",
			error="corrupted image",			
			status="failed"
		)

	
@app.route("/search", methods=['GET', 'POST'])		
def search():
	return "search a face within given video"
	
@app.route("/emotions", methods=['GET', 'POST'])		
def emotions():
	return "analyze emotions within give video"
	
@app.route("/detect", methods=['GET', 'POST'])		
def detect():
	return "detect for attributes(smile, blink within given video"

if __name__ == "__main__":
    #start_runner()
    app.run()
	
