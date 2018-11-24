import bz2
import os
from model import create_model
from urllib.request import urlopen
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from align import AlignDlib
from sklearn.metrics import f1_score, accuracy_score


def download_landmarks(dst_file):
	url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
	decompressor = bz2.BZ2Decompressor()
    
	with urlopen(url) as src, open(dst_file, 'wb') as dst:
		data = src.read(1024)
		while len(data) > 0:
			dst.write(decompressor.decompress(data))
			data = src.read(1024)
			dst_dir = 'models'
			
def load_metadata(path):
	metadata = []
	for i in os.listdir(path):
		for f in os.listdir(os.path.join(path, i)):
			# Check file extension. Allow only jpg/jpeg' files.
			ext = os.path.splitext(f)[1]
			if ext == '.jpg' or ext == '.jpeg' or ext =='.png':
				metadata.append(IdentityMetadata(path, i, f))
	return np.array(metadata)
	
def align_image(img):
	return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
							landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


class IdentityMetadata():
	def __init__(self, base, name, file):
		# dataset base directory
		self.base = base
		# identity name
		self.name = name
		# image file name
		self.file = file

	def __repr__(self):
		return self.image_path()

	def image_path(self):
		return os.path.join(self.base, self.name, self.file) 
		
dst_dir = 'models'
dst_file = os.path.join(dst_dir, 'landmarks.dat')	
alignment = AlignDlib('models/landmarks.dat')	
def init():
	if not os.path.exists(dst_file):
		os.makedirs(dst_dir)
		download_landmarks(dst_file)
	else:
		print("landmark_file already downloaded")
		
	model = create_model()
	model.load_weights('weights/nn4.small2.v1.h5')
	data = load_metadata('images')
	
	return [model, data]