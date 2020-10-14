import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
import random
import glob
from sklearn.utils import class_weight

from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import np_utils
from keras.optimizers import SGD, adam
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import optimizers

import tensorflow as tf

# Testing the mobile app

NAME = "MobileApp_7_Classes"
NUM_CLASSES = 7
EPOCHS = 15
BATCH_SIZE = 32
CLASSES = [
'Grape___Black_rot'
,'Corn_(maize)___healthy'
,'Apple___healthy'
,'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
,'Squash___Powdery_mildew'
,'Orange___Haunglongbing_(Citrus_greening)'
,'Peach___Bacterial_spot']
#,'Tomato___healthy' 
#,'Potato___healthy'
#,'Apple___Black_rot']

print("Running: " + NAME)

def print_training_data(acc, val_acc, loss, val_loss):
  print('Training Accuracy:\n', acc) 
  print('Validation Accuracy:\n', val_acc) 
  print('Training Loss:\n', loss) 
  print('Validation Loss:\n', val_loss) 

def write_data_to_file(acc, val_acc, loss, val_loss, filename):
  # Write data to .csv file
  # Note: Clear data.csv each time! After clearing, add '0' to make it non-empty
  # filename = 'data.csv'
  open(filename, 'w+').close() # Clear file before writing to it (and create if nonexistent)
  with open(filename, 'w') as f:
    f.write('0') # Add a value
  f.close()
  print('Writing data to .csv file...')
  data = pd.read_csv(filename, 'w') 
  data.insert(0,"Training Acc", acc)
  data.insert(1,"Training Loss", val_acc)
  data.insert(2,"Validation Acc", loss)
  data.insert(3,"Validation Loss", val_loss)
  data.to_csv(filename)
  print('Finished writing data!')

def plot_graphs(acc, val_acc, loss, val_loss):
  # Plot results
  print("Starting plotting...") 
  epochs = range(1, len(acc)+1)
  plt.plot(epochs, acc, 'bo', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title('Training and Validation accuracy')
  plt.legend()
  plt.savefig('plots/Plants_TrainingAcc.png')
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and Validation loss')
  plt.legend()
  plt.ylim([0,4]) # Loss should not increase beyond 4! 
  plt.savefig('plots/Plants_TrainingLoss.png')
  plt.figure()
  plt.show()
  print('Finished plotting!')

def get_inceptionV3():
  # base_model = tf.keras.applications.MobileNetV2(include_top = False, weights = 'imagenet', input_shape=(224, 224, 3))
  base_model = tf.keras.applications.InceptionV3(include_top = False, weights = 'imagenet', input_shape=(224, 224, 3))
  # base_model = tf.keras.applications.ResNet50(include_top = False, weights = 'imagenet', input_shape=(224, 224, 3))

  model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='prediction')
  ])
  
  model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.005), metrics=['accuracy', 'top_k_categorical_accuracy'])
  return model

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

  # FIXME: REMOVE!!!
  # classes = ['Grape___Black_rot', 'Corn_(maize)___healthy']
  classes = CLASSES
  print("classes: ", classes)

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
    print("Total number of image files: ", num_plants)

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
  train_y = np_utils.to_categorical(train_y, NUM_CLASSES)
  val_y = np_utils.to_categorical(val_y, NUM_CLASSES)
  test_y = np_utils.to_categorical(test_y, NUM_CLASSES)
  print("Successfully loaded data!")

  return train_x, val_x, test_x, train_y, val_y, test_y, classweight, classes

def write_labels_to_file(filename):
  with open(filename, 'w+') as f:
    for leaf_class in CLASSES:
      f.write(leaf_class+'\n')

def convert_model_to_tflite(model):
  # Save model
  print("\nSAVING MODEL...")
  saved_model_dir = 'save/fine_tuning'
  tf.saved_model.save(model, saved_model_dir)

  print("SAVED MODEL! NOW CONVERTING...")
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  tflite_model = converter.convert()

  print("CONVERTED MODEL! NOW WRITING TO FILE...")
  with open(NAME + '.tflite', 'wb') as f:
    f.write(tflite_model)
  print("WROTE MODEL TO FILE!")

# ------------------------------ Main ------------------------------ #
# Load model
model = get_inceptionV3()
print(model.summary())
print("Loaded " + NAME + " model successfully!")

# This part will take a while
data_dir = '/scratch/user/skumar55/plantvillage_deeplearning_paper_dataset/raw/color/'
train_x, val_x, test_x, train_y, val_y, test_y, classweight, classes = load_plantvillage(seed=7, root_dir=data_dir)
print(train_x.shape, val_x.shape, test_x.shape)   
print(train_y.shape, val_y.shape, test_y.shape)

# Train model
es = EarlyStopping(patience=10)
model_checkpoint = ModelCheckpoint(filepath='saved_model_weights/weights.{epoch:02d}-{val_loss:.2f}'+NAME+'.h5', monitor='val_loss', verbose=1, save_weights_only=False, period=1)
history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_x, val_y), callbacks=[es, model_checkpoint], class_weight=classweight, steps_per_epoch=len(train_x)/BATCH_SIZE)

convert_model_to_tflite(model)
write_labels_to_file('labels.txt')

# Save training parameters
print('Obtaining training data...')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Print data
print_training_data(acc, val_acc, loss, val_loss)
write_data_to_file(acc, val_acc, loss, val_loss, 'data_'+NAME+'.csv')
plot_graphs(acc, val_acc, loss, val_loss)

# Get testing results 
test_results = model.evaluate(test_x, test_y)
print("Printing test results...")
for metric, name, in zip(test_results, ['loss', 'acc', 'top 5 acc']):
  print(name, metric)
print("Model finished!")
