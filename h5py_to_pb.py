import tensorflow as tf
import argparse
import uuid
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import shutil
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert Keras .h5 file to Tensorflow .pb frozen model.")
parser.add_argument("--h5", type=str, default="model.h5", help="Keras .h5 file (input)")
parser.add_argument("--pb", type=str, default="./saved_model", help="Path to save Tensorflow frozen model (output)")
args = parser.parse_args()

K.set_learning_phase(0)

model = keras.models.load_model(args.h5)
model_output = model.output.op.name

tf.keras.models.save_model(
    model,
    args.pb,
    overwrite=True,
    include_optimizer=True,
    save_format='tf',
    signatures=None,
    options=None
)


#model = tf.keras.models.load_model("")
model.summary()