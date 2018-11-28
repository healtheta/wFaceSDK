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

from api import create as CREATE
from api import update as UPDATE
from api import train as TRAIN
from api import predict as PREDICT	
from api import clear as CLEAR
from api import recheck as CONFIRM

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

@app.route("/create", methods=['GET', 'POST'])	
def create():
	resp = CREATE()
	return jsonify(
		id =resp,
		error="none",			
		status="success"
	)

@app.route("/update", methods=['GET', 'POST'])		
def update():
	content = request.json
	id = content['id']
	image = content['image']
	#format, imgstr = image.split(';base64,') 	
	#ext = format.split('/')[-1] 	
	#image = base64.b64decode(imgstr)
	#n = random.randint(0,100000);
	#file_name = str(n) + "." + ext	
	x = UPDATE(id, image)
	if x:
		return jsonify(
			id =id,
			error="none",			
			status="success"
		)	
	else:
		return jsonify(
			id =id,
			error="none",			
			status="fail"
		)			
	
@app.route("/train", methods=['GET', 'POST'])		
def train():
	TRAIN()
	CLEAR()
	return jsonify(
		error="none",			
		status="success"
	)	
	
@app.route("/predict", methods=['GET', 'POST'])		
def predict():
	content = request.json
	image = content['image']
	#format, imgstr = image.split(';base64,') 	
	#ext = format.split('/')[-1] 	
	#image = base64.b64decode(imgstr)
	#n = random.randint(0,100000);
	#file_name = str(n) + "." + ext	
	#full_filename = os.path.join("tmp", file_name)
	#with open(full_filename,"wb") as fo:
	#	fo.write(image)
		
	id = PREDICT(image)
	CLEAR()	
	full_filename = 'pragash'
	if CONFIRM(full_filename, id):
		return jsonify(
			id =id,
			error="none",			
			status="success"
		)	
	else:
		return jsonify(
			id ='unknown',
			error="none",			
			status="success"
		)			
	
	
	
	

if __name__ == "__main__":
    #start_runner()
    app.run()
	
