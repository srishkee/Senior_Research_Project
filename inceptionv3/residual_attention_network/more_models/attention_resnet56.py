# TEST FOR GPU:
from __future__ import absolute_import, division, print_function, unicode_literals
print("\nRUNNING: ATTENTION_resnet50.py script!\n")
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Import necessary modules
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import random
import numpy as np
import glob
from sklearn.utils import class_weight
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import keras
import math
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from models import AttentionResNetCifar10, AttentionResNet92, AttentionResNet56

import matplotlib.pyplot as plt

# from toda_okura_inceptionv3 import load_plantvillage

print("\nRUNNING: ATTENTION_resnet50.py script!\n")

# Removed step decay functions

# def step_decay(epoch):
#   initial_lrate = 0.1
#   drop = 0.5 # How much to divide the learning rate by
#   epochs_drop = 10# FIXME!!! Divide learning rate at these many epochs 
#   lrate = initial_lrate * math.pow(drop, math.floor( (1+epoch) / epochs_drop ))
#   return lrate

# class LossHistory(keras.callbacks.Callback):
#   def on_train_begin(self, logs={}):
#     self.losses=[]
#     self.lr=[]

#   def on_epoch_end(self, batch, logs={}):
#     self.losses.append(logs.get('loss'))
#     self.lr.append(step_decay(len(self.losses)))


# Load plant data 
def load_plantvillage(seed=None, root_dir=None):
  if(root_dir==None):
    print("Please enter a valid data directory!")
    return
  
  random.seed(seed)

  def read_from_paths(paths):
    x=[]
    for path in paths:
      # print('path: ', path)
      img = load_img(path, target_size=(224,224))
      img = img_to_array(img)
      x.append(img)
    return x

  classes = os.listdir(root_dir)
  classes = sorted(classes)

  train_path = []
  val_path = []
  test_path = []

  train_x, train_y = [],[]
  val_x, val_y = [],[]
  test_x, test_y = [],[]

  # Read paths and split data into 6:2:2 dataset
  for i, _class in enumerate(classes):
    paths = glob.glob(os.path.join(root_dir, _class, "*"))
    paths = [n for n in paths if n.endswith(".JPG") or n.endswith(".jpg")]
    random.shuffle(paths)
    num_plants = len(paths)
    # print("num_plants: ", num_plants)

    train_path.extend(paths[:int(num_plants*0.6)])
    train_y.extend([i]*int(num_plants*0.6))

    val_path.extend(paths[int(num_plants*0.6):int(num_plants*0.8)])
    val_y.extend([i]*len(paths[int(num_plants*0.6):int(num_plants*0.8)]))

    test_path.extend(paths[int(num_plants*0.8):])
    test_y.extend([i]*len(paths[int(num_plants*0.8):]))

  print("Loading images...")

  train_x = read_from_paths(train_path)
  print("Loaded all training images!")
  val_x = read_from_paths(val_path)
  print("Loaded all validation images!")
  test_x = read_from_paths(test_path)
  print("Loaded all testing images!")

  # Convert all to numpy
  train_x = np.array(train_x)/255.
  train_y = np.array(train_y)
  val_x = np.array(val_x)/255.
  val_y = np.array(val_y)
  test_x = np.array(test_x)/255.
  test_y = np.array(test_y)

  # Calculate class weight
  classweight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)

  # Convert to categorical (~1-hot encoding)
  train_y = np_utils.to_categorical(train_y, 38)
  val_y = np_utils.to_categorical(val_y, 38)
  test_y = np_utils.to_categorical(test_y, 38)
  print("Successfully loaded data!")

  return train_x, val_x, test_x, train_y, val_y, test_y, classweight, classes

# ------------------------------ Main ------------------------------ #

# build a model
model = AttentionResNet56(n_classes=38)

# Load data
data_dir = '/scratch/user/skumar55/plantvillage_deeplearning_paper_dataset/raw/color/'
# data_dir = '/mnt/scratch/skumar55_data/plant_data/'
x_train, x_val, x_test, y_train, y_val, y_test, classweight, classes = load_plantvillage(seed=7, root_dir=data_dir)

# define generators for training and validation data
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    # rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

val_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
train_datagen.fit(x_train)
val_datagen.fit(x_val)
test_datagen.fit(x_test)

# Callbacks
# loss_history = LossHistory()
# lrate = LearningRateScheduler(step_decay)
lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=10e-7, epsilon=0.01, verbose=1)
early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=7, verbose=1)
model_checkpoint = ModelCheckpoint(filepath='saved_model_weights/attention_resnet56.h5', monitor='val_loss', verbose=1, save_weights_only=False, period=1)
callbacks = [lr_reducer, early_stopper, model_checkpoint] # Removed model_checkpoint from callbacks list

# define loss, metrics, optimizer
model.compile(keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy']) # Note: lr = 1/10th that of paper!

# fits the model on batches with real-time data augmentation
batch_size = 64

history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train)//batch_size, epochs=100,
                    validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size), 
                    validation_steps=len(x_val)//batch_size,
                    initial_epoch=0, callbacks=callbacks) # Removed callbacks
print ("Successfully fit model to data!")

model.save('attention_resnet56.h5')
print("Saved model successfully!")

model.evaluate_generator(test_datagen.flow(x_test, y_test), steps=len(x_test)/32, use_multiprocessing=True)
print ("Evaluating model!")



# Plotting results
import matplotlib.pyplot as plt

val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
# loss = history.history['loss']

print('\nloss: ', loss)
print('val_loss: ', val_loss)
print('accuracy: ', accuracy)
print('val_accuracy: ', val_accuracy)
# print('\nlearning_rate: ', loss_history.lr)
# print('losses: ', loss_history.losses)

epochs = range(1, len(accuracy)+1)

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.savefig('TrainingAccuracy')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.savefig('TrainingLoss')
plt.figure()
plt.show()


