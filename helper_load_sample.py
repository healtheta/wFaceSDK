from init import init
from init import align_image
from init import load_metadata
from init import init_model
from helper import load_image
import numpy as np
import json

from db import connect as DB_CONNECT
from db import add as DB_ADD
from db import list as DB_LIST
from db import close as DB_CLOSE

DB_CONNECT()
model , data = init()
embedded = np.zeros((data.shape[0], 128))
for i, m in enumerate(data):
	img = load_image(m.image_path())
	img = align_image(img)
	# scale RGB values to interval [0,1]
	img = (img / 255.).astype(np.float32)
	# obtain embedding vector for image
	embedded[i] = model.predict(np.expand_dims(img, axis=0))[0]
	embedded = embedded[i].tolist()
	DB_ADD(m.name,json.dumps(embedded))
	
DB_CLOSE()


