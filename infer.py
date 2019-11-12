import argparse
import json

import numpy as np
import requests
from keras.applications import inception_v3
from keras.preprocessing import image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
img = image.img_to_array(image.load_img(image_path, target_size=(160, 160))) / 255.


img = img.astype('float16')

payload = {
    "instances": [{'input_1': img.tolist()}]
}

r = requests.post('http://localhost:8501/v1/models/sla:predict', json=payload)
print(r.content)
#pred = json.loads(r.content.decode('utf-8',errors="ignore").replace("\\n", "\n"))

#print(pred)
#print(r.json())