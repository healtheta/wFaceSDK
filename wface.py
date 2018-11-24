import requests
import threading
import time
import face_recognition
import base64
import random
import os
import json
import numpy as np

#encode an image in to facial points
def wface_encode(image):
	tmp = None;
	try:
		tmp = face_recognition.load_image_file(image)
	except FileNotFoundError:
		print("file not found")
		return tmp
	face_encoding = None;
	try:
		face_encoding = face_recognition.face_encodings(tmp)[0]
	except IndexError:
		print("corrrpted image")
		return face_encoding
		
	return face_encoding;
	
#do a match between a known encoding and unknown encoding
def wface_compare(known, unknown):
	known_list = []
	known_list.append(known)
	list_ret = face_recognition.compare_faces(known_list, unknown)
	return list_ret[0];
	
def wface_search_from_list(list, unknown):
	list_ret = face_recognition.compare_faces(list, unknown)
	
	index = 0
	for elem in list_ret:
		if elem == True:
			return index;
		index = index + 1;
		
def wface_find_emotion(image):
	print("finding emotions")
	
def wface_find_liveliness(video):
	print("finding liveliness")
	
def search_video(person):
	print("search video for a person")
	
	
	