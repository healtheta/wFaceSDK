import uuid
import os
from init import init
from init import align_image
from init import load_metadata
from helper import load_image
import bz2
from model import create_model
from urllib.request import urlopen
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from align import AlignDlib
from sklearn.metrics import f1_score, accuracy_score
from joblib import dump, load
from model import create_model
from keras import backend as K
import face_recognition
import time

def create():
	x = uuid.uuid1()
	x = str(x)
	full_filename = os.path.join("images", x)
	if os.path.exists(full_filename):
		create()
	else:
		os.mkdir(full_filename)
		return x
	
def update(eId, data, filename):
	full_filename = os.path.join("images", eId, filename)
	fout = open(full_filename, 'wb+')		
	with open(full_filename,"wb") as fo:
		fo.write(data)
	fout.close()	
	return True;
	
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import numpy as np
def train():
	begin = time.clock()
	model , data = init()
	embedded = np.zeros((data.shape[0], 128))
	start = time.clock()
	for i, m in enumerate(data):
		img = load_image(m.image_path())
		img = align_image(img)
		# scale RGB values to interval [0,1]
		img = (img / 255.).astype(np.float32)
		# obtain embedding vector for image
		embedded[i] = model.predict(np.expand_dims(img, axis=0))[0]
		
	targets = np.array([m.name for m in data])
	print(time.clock() - start)
	encoder = LabelEncoder()
	encoder.fit(targets)

	# Numerical encoding of identities
	y = encoder.transform(targets)

	train_idx = np.arange(data.shape[0]) % 2 != 0
	test_idx = np.arange(data.shape[0]) % 2 == 0
	start = time.clock()
	# 50 train examples of 10 identities (5 examples each)
	X_train = embedded[train_idx]
	# 50 test examples of 10 identities (5 examples each)
	X_test = embedded[test_idx]

	y_train = y[train_idx]
	y_test = y[test_idx]
	
	knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
	svc = LinearSVC()

	knn.fit(X_train, y_train)
	svc.fit(X_train, y_train)
	print(time.clock() - start)
	acc_knn = accuracy_score(y_test, knn.predict(X_test))
	acc_svc = accuracy_score(y_test, svc.predict(X_test))
	dump(svc, 'models/svc.joblib')
	np.save('models/classes.npy', encoder.classes_)
	print(time.clock() - begin)
		
	
	
def predict(image):
	svc = load('models/svc.joblib')
	encoder = LabelEncoder()
	encoder.classes_ = np.load('models/classes.npy')
	img = load_image(image)
	img = align_image(img)
	# scale RGB values to interval [0,1]
	img = (img / 255.).astype(np.float32)
	# obtain embedding vector for image
	model = create_model()
	model.load_weights('weights/nn4.small2.v1.h5')
	embedded = model.predict(np.expand_dims(img, axis=0))[0]	
	prediction = svc.predict([embedded])
	identity = encoder.inverse_transform(prediction)[0]		
	return identity
	
def recheck(image, id):
	test = face_recognition.load_image_file(image)
	full_filename = os.path.join("images", id)
	data = load_folder(full_filename)
	
	predicted_encodings = []
	for img in data:
		predict_img = face_recognition.load_image_file(img)
		predicted_encod = face_recognition.face_encodings(predict_img)[0]
		predicted_encodings.append(predicted_encod)

	test_encode = face_recognition.face_encodings(test)[0]
	face_distances = face_recognition.face_distance(predicted_encodings, test_encode)
	
	total = 0;
	i = 0;
	for distance in face_distances:
		total = total + distance;
		i = i + 1;
		
	avg = total / i;
	
	if avg < 0.5:
		return True;
	else:
		return False;
	
	
	
def load_folder(path):
	metadata = []
	for i in os.listdir(path):
		full = os.path.join(path, i)
		ext = os.path.splitext(full)[1]
		if ext == '.jpg' or ext == '.jpeg' or ext =='.png':
			metadata.append(full)
			
			
		'''for f in os.listdir(os.path.join(path, i)):
			# Check file extension. Allow only jpg/jpeg' files.
			ext = os.path.splitext(f)[1]
			if ext == '.jpg' or ext == '.jpeg' or ext =='.png':
				metadata.append(IdentityMetadata(path, i, f))'''
	return np.array(metadata)
	
def clear():
	K.clear_session()
	
train()