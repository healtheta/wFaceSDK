from keras.models import model_from_json
from keras.optimizers import SGD
import numpy as np
from time import sleep
from scipy.ndimage import zoom
import cv2


model = model_from_json(open('./models/Face_model_architecture.json').read())
#model.load_weights('_model_weights.h5')
model.load_weights('./models/Face_model_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


def wem_extract_face_features(gray, detected_face, offset_coefficients):
		(x, y, w, h) = detected_face
		#print x , y, w ,h
		horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
		vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
	

		extracted_face = gray[y+vertical_offset:y+h, 
						  x+horizontal_offset:x-horizontal_offset+w]
		#print extracted_face.shape
		new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0], 
											   48. / extracted_face.shape[1]))
		new_extracted_face = new_extracted_face.astype(np.float32)
		new_extracted_face /= float(new_extracted_face.max())
		return new_extracted_face

def wem_detect_face(frame):
		cascPath = "./models/haarcascade_frontalface_default.xml"
		faceCascade = cv2.CascadeClassifier(cascPath)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		detected_faces = faceCascade.detectMultiScale(
				gray,
				scaleFactor=1.1,
				minNeighbors=6,
				minSize=(48, 48),
				flags=cv2.CASCADE_SCALE_IMAGE
			)
		return gray, detected_faces
		
		
def wem_detect_emotion(frame, emotion=None):
	# detect faces
	gray, detected_faces = wem_detect_face(frame)
	# predict output
	for face in detected_faces:
		# extract features
		extracted_face = wem_extract_face_features(gray, face, (0.075, 0.05)) #(0.075, 0.05)
		prediction_result = model.predict_classes(extracted_face.reshape(1,48,48,1))
		if prediction_result == 3:
			return "Happy"
		elif prediction_result == 0:
			return "Angry"
		elif prediction_result == 1:
			return "Disgust"		
		elif prediction_result == 2:
			return "Fear"		
		elif prediction_result == 4:
			return "Sad"			
		elif prediction_result == 5:
			return "Suprise"		
		else :
			return "Neutral"	
		
