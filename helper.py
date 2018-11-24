#face-alignment
import cv2

def load_image(path):
	img = cv2.imread(path, 1)
	# OpenCV loads images with color channels
	# in BGR order. So we need to reverse them
	return img[...,::-1]