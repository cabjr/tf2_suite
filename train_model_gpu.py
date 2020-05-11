import tensorflow as tf
import glob, os, argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import date, time
import keras
import datetime
print(tf.__version__)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction=0.9 # don't hog all vRAM
config.operation_timeout_in_ms=15000   # terminate on long hangs
sess = InteractiveSession("", config=config)

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

BATCH_SIZE = 8
FC_LAYERS = [256, 256]

def scheduler(epoch):
  if epoch < 10:
    return 0.0001
  else:
    return 0.0001 * tf.math.exp(0.1 * (10 - epoch))

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
  for layer in base_model.layers:
    layer.trainable = False
  x = base_model.output
  x = tf.keras.layers.Flatten()(x)
  for fc in fc_layers:
    x = tf.keras.layers.Dense(fc, activation='relu')(x) # New FC layer, random init
    x = tf.keras.layers.Dropout(dropout)(x)
  predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x) # New softmax layer
  finetune_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
  return finetune_model

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
parser.add_argument('--mode', type=str, default="train", help='Mode "train" or "predict"')
parser.add_argument('--dataset', type=str, default=None, help='dataset directory containing folder "train" and "val"')
parser.add_argument('--model', type=str, default="InceptionV3", help='Model to be used in the training or predict ("InceptionV3", "MobileNet", "MobileNetV2", "InceptionResNetV2", "ResNet", "NasNet", "VGG16", "VGG19", "Xception")')
parser.add_argument('--optimizer', type=str, default="Adam", help='Optimizer to be used on the training\nOptions:\nAdam, Adagrad, RMSProp, SGD\n')
parser.add_argument('--image_size', type=int, default=160, help='Image size ("96", "144", "160", "224") to be processed')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to be used on the training')
parser.add_argument('--epoch_steps', type=int, default=500, help='Number of steps per epoch')
parser.add_argument('--val_steps', type=int, default=50, help='Number of steps per epoch')
parser.add_argument('--num_epoch', type=int, default=10, help='Number epochs to train')
parser.add_argument('--retrain_hdf5', type=str, default=None, help='Specify the path to hdf5 file to continue the training')
parser.add_argument('--show_chart', type=str2bool, default=False, help='Show chart after training? (true, false)')
parser.add_argument('--logs_dir', type=str, default=None, help='Path to save the logs of the training')
parser.add_argument('--output_dir', type=str, default=".", help='Path to save the checkpoints and model')
parser.add_argument('--validate_every', type=int, default=10, help='Number of epochs until validation')
parser.add_argument('--lr_decay', type=str2bool, default=False, help='Use learning rate decay?')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
parser.add_argument('--validate_save_every', type=int, default=3, help='Validate and Save Checkpoints every X number of epochs')
parser.add_argument('--finetune', type=str2bool, default=False, help='Define if the retrain_hdf5 will be used to finetune the network')
args = parser.parse_args()

def train():
  global IMAGE_SIZE, BATCH_SIZE
  if (args.mode.lower() == 'train'):
    if (args.dataset== None):
      print("You must specify a path to the dataset")
      return
    if ('train' in os.listdir(args.dataset) and 'val' in os.listdir(args.dataset)) == False:
      print("The dataset folder must contain 'train' and 'val' folders")
      return
    train_filenames, train_labels = read_datasets(args.dataset+"/train")
    val_filenames, val_labels = read_datasets(args.dataset+"/val")
    train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(train_labels)))
    val_data = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames), tf.constant(val_labels)))

    IMAGE_SIZE = args.image_size
    BATCH_SIZE = args.batch_size
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

    # MODELS
    if (args.model.lower() == 'inceptionv3'):   
      base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif (args.model.lower() == 'inceptionresnetv2'):
      base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif (args.model.lower() == 'mobilenet'):
      base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif (args.model.lower() == 'mobilenetv2'):
      base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif (args.model.lower() == 'resnet'):
      base_model = tf.keras.applications.ResNet(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif (args.model.lower() == 'nasnet'):
      base_model = tf.keras.applications.NasNet(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif (args.model.lower() == 'vgg16'):
      base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif (args.model.lower() == 'vgg19'):
      base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif (args.model.lower() == 'xception'):
      base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    else:
      print('You must specify a model architecture ("InceptionV3", "MobileNet", "MobileNetV2", "InceptionResNetV2", "ResNet", "NasNet", "VGG16", "VGG19", "Xception")')
      return
    base_model = build_finetune_model(base_model, args.dropout, FC_LAYERS, len(os.listdir(args.dataset+"/train")))
    train_data = (train_data.map(_parse_fn).shuffle(buffer_size=4000).batch(BATCH_SIZE))
    val_data = (val_data.map(_parse_fn).shuffle(buffer_size=800).batch(BATCH_SIZE))

    #OPTIMIZERS
    lr = args.lr
    if args.optimizer.lower()=="adam":
      opt = tf.keras.optimizers.Adam(lr=lr)
    elif args.optimizer.lower()=="adagrad":
      opt = tf.keras.optimizers.Adagrad(lr=lr)
    elif args.optimizer.lower()=="rmsprop":
      opt = tf.keras.optimizers.RMSprop(lr=lr)
    elif args.optimizer.lower()=="sgd":
      opt = tf.keras.optimizers.SGD(lr=lr)
    else:
      print("The given optimizer is not supported")
      return
    
    callback_list = []

    dt = datetime.datetime.now()
    outDir = args.output_dir + "/ckpt/%s" % (str(dt.day)+"_"+str(dt.month)+"_"+str(dt.hour)+"-"+str(dt.minute)+"/")
    if not os.path.exists(outDir):
      os.makedirs(outDir)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(outDir+"/model_"+args.model+"_-{epoch:02d}.hdf5", monitor='val_accuracy', verbose=1,save_best_only=False, save_weights_only=False, save_frequency='epoch')
    if (args.finetune == False):
      model = base_model
    else:
      model = tf.keras.models.load_model(args.retrain_hdf5)
      model.load_weights(args.retrain_hdf5)
      print(model.layers)
      model.summary()
      #x = model.get_layer('flatten').output
      x = model.get_layer(model.layers[-5].name).output
      x = tf.keras.layers.Flatten()(x)
      for fc in FC_LAYERS:
        x = tf.keras.layers.Dense(fc, activation='relu')(x)
        x = tf.keras.layers.Dropout(args.dropout)(x)
      predictions = tf.keras.layers.Dense(len(os.listdir(args.dataset+"/train")), activation='softmax')(x)
      model = tf.keras.Model(inputs=model.input, outputs=predictions)
      print(5*"============")
      print("Modelo carregado sem problemas")
      print(5*"============")
      model.summary()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    num_epochs = int(args.num_epoch)
    steps_per_epoch = int(args.epoch_steps)
    val_steps = int(args.val_steps)
    callback_list.append(checkpoint_callback)
    if (args.lr_decay):
      callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
      callback_list.append(callback)
    if (args.retrain_hdf5 is not None and args.finetune == False):
      model.load_weights(args.retrain_hdf5)
      print("Restoring weights from file: "+ args.retrain_hdf5)
    history = model.fit(train_data.repeat(), epochs=num_epochs, steps_per_epoch = steps_per_epoch, validation_data=val_data.repeat(), validation_steps=val_steps, callbacks=callback_list, validation_freq=1)
    #SAVE
    model.save_weights(args.output_dir+"/weights_model_"+args.model+".hdf5")
    model.save(args.output_dir+"/model_"+args.model+".h5")

    if (args.show_chart):
      #SHOW STATISTICS OF THE MODEL
      acc = history.history['accuracy']
      val_acc = history.history['val_accuracy']
      loss = history.history['loss']
      val_loss = history.history['val_loss']
      plt.figure(figsize=(8, 8))
      plt.subplot(2, 1, 1)
      plt.plot(acc, label='Training Accuracy')
      plt.plot(val_acc, label='Validation Accuracy')
      plt.legend(loc='lower right')
      plt.ylabel('Accuracy')
      plt.title('Training and Validation Accuracy')
      plt.subplot(2, 1, 2)
      plt.plot(loss, label='Training Loss')
      plt.plot(val_loss, label='Validation Loss')
      plt.legend(loc='upper right')
      plt.ylabel('Cross Entropy')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')
      plt.show()
 
train()
