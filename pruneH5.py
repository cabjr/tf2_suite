import numpy as np
import argparse, glob, os
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_datasets(dtset_dir):
    lbls = []
    filenames = []
    for folder in os.listdir(dtset_dir):
        indexOfClass = os.listdir(dtset_dir).index(folder)
        this_hot = tf.keras.utils.to_categorical(indexOfClass, num_classes=len(os.listdir(dtset_dir)), dtype='float32')
        lbls += [this_hot for i in range(len(glob.glob(dtset_dir+"/"+folder+"/*.jpg")))]
        filenames += [f.replace("\\","/") for f in glob.glob(dtset_dir+"/"+folder+"/*.jpg")]
    return filenames, lbls
    
# Function to load and preprocess each image
def _parse_fn(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    #image_decoded = image_decoded[:,:,:3]
    image_normalized = (tf.cast(image_decoded, tf.float32)/255) - 1
    image_resized = tf.image.resize(image_normalized, (IMAGE_SIZE, IMAGE_SIZE))
    return image_resized, label

parser = argparse.ArgumentParser()
parser.add_argument('--h5', type=str, default=None, help='Specify the path to hdf5 file to continue the training')
parser.add_argument('--dataset', type=str, default=None, help='dataset directory containing folder "train" and "val"')
parser.add_argument('--image_size', type=int, default=224, help='Image size ("96", "144", "160", "224") to be processed')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to be used on the training')
parser.add_argument('--tflite', type=str2bool, default=False, help='Generate tflite model?')
args = parser.parse_args()


#loaded_model = tf.keras.models.load_model("D:/dataset/longe.hdf5")
loaded_model = tf.keras.models.load_model(args.h5)
"""
epochs = 4
end_step = np.ceil(1.0 * 600 / args.batch_size).astype(np.int32) * epochs
print(end_step)

new_pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.90, begin_step=0, end_step=end_step, frequency=100)}

new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
new_pruned_model.summary()


#dataset
train_filenames, train_labels = read_datasets(args.dataset+"/train")
val_filenames, val_labels = read_datasets(args.dataset+"/val")
train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(train_labels)))
val_data = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames), tf.constant(val_labels)))

IMAGE_SIZE = args.image_size
BATCH_SIZE = args.batch_size

train_data = (train_data.map(_parse_fn, num_parallel_calls=2).shuffle(buffer_size=len(glob.glob(args.dataset+"/train/*/*.jpg"))).batch(BATCH_SIZE))
val_data = (val_data.map(_parse_fn, num_parallel_calls=2).shuffle(buffer_size=len(glob.glob(args.dataset+"/val/*/*.jpg"))).batch(BATCH_SIZE))


new_pruned_model.compile( loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

callbacks = [ sparsity.UpdatePruningStep(),sparsity.PruningSummaries(log_dir=logdir, profile_batch=0) ]
new_pruned_model.fit(x_train, y_train, batch_size=args.batch_size, epochs=epochs, verbose=1, callbacks=callbacks, validation_data=(x_test, y_test))
score = new_pruned_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

final_model = sparsity.strip_pruning(pruned_model)
final_model.summary()

model = load_model(final_model)
import numpy as np
"""
for i, w in enumerate(loaded_model.get_weights()):
    print("{} -- Total:{}, Zeros: {:.2f}%".format(loaded_model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100))

import tempfile
import zipfile

_, new_pruned_keras_file = tempfile.mkstemp(".h5")
print("Saving pruned model to: ", new_pruned_keras_file)
tf.keras.models.save_model(loaded_model, new_pruned_keras_file, include_optimizer=False)

# Zip the .h5 model file
_, zip3 = tempfile.mkstemp(".zip")
with zipfile.ZipFile(zip3, "w", compression=zipfile.ZIP_DEFLATED) as f:
    f.write(new_pruned_keras_file)

print("Size of the pruned model before compression: %.2f Mb" % (os.path.getsize(new_pruned_keras_file) / float(2 ** 20)))
print("Size of the pruned model after compression: %.2f Mb" % (os.path.getsize(zip3) / float(2 ** 20)))

tflite_model_file = "./converted.tflite"
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(new_pruned_keras_file)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)