import tensorflow as tf
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--h5', type=str, default=None, help='Mode "train" or "predict"')
parser.add_argument('--size', type=int, default=160, help='Input size to the model')
parser.add_argument('-images','--images', nargs='+', help='<Required> One or more paths to the images that will be predicted', required=True)
args = parser.parse_args()

def _parse_fn(filename):
  image_string = tf.io.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  #image_decoded = image_decoded[:,:,:3]
  image_normalized = (tf.cast(image_decoded, tf.float32)/255) - 1
  image_resized = tf.image.resize(image_normalized, (args.size, args.size))
  return image_resized


if args.h5 is not None:
    model = tf.keras.models.load_model(args.h5)
    #model.summary()
    for item in args.images:
        x = _parse_fn(item)
        img_rgb = cv2.imread(item)
        img_rgb = cv2.resize(img_rgb,(args.size,args.size),3)  # resize
        img_rgb = np.array(img_rgb).astype(np.float32)/255.0  # scaling
        img_rgb = np.expand_dims(img_rgb, axis=0)  # expand dimension
        result = model.predict(img_rgb, batch_size=None, verbose=0, steps=None, callbacks=None,  max_queue_size=10, workers=1,use_multiprocessing=False)
        print(item)
        print([ '%.4f' % elem for elem in result[0]])