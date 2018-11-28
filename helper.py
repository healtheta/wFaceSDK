#face-alignment
import cv2
import base64
from PIL import Image
import cv2

try:
    from BytesIO  import BytesIO 
except ImportError:
    from io import BytesIO 
import numpy as np

def load_image(path):
	img = cv2.imread(path, 1)
	# OpenCV loads images with color channels
	# in BGR order. So we need to reverse them
	return img[...,::-1]
	
def load_image_from_memory(base64_string):
	sbuf = BytesIO ()
	sbuf.write(base64.b64decode(base64_string))
	sbuf.seek(0)
	pimg = Image.open(sbuf)
	img =  cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
	img = img[...,::-1]
	return img