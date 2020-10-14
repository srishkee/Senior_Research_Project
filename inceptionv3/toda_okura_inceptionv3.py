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

# Note: This file contains the original author's code (Toda-Okura) for building&training the InceptionV3 model.
# Reaches the expected >95% accuracy.

print("RUNNING: COLOR InceptionV3!")

data_dir = '/scratch/user/skumar55/plantvillage_deeplearning_paper_dataset/raw/color/'
print("Plant categories: ", os.listdir(data_dir))

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
  base_model = InceptionV3(include_top = False, weights = None, input_shape=(224, 224, 3))
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(38, name='preprediction')(x)
  predictions = Activation('softmax', name='prediction')(x)
  model = Model(inputs=base_model.input, outputs = predictions)
  model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.005), metrics=['accuracy', 'top_k_categorical_accuracy'])
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
# This part will take a while
train_x, val_x, test_x, train_y, val_y, test_y, classweight, classes = load_plantvillage(seed=7, root_dir=data_dir)

print(train_x.shape, val_x.shape, test_x.shape)   
print(train_y.shape, val_y.shape, test_y.shape)

model = get_inceptionV3()
es = EarlyStopping(patience=10)
history = model.fit(train_x, train_y, epochs=50, batch_size=32, validation_data=(val_x, val_y), callbacks=[es], class_weight=classweight)

# Save model 
print ("Saving model...")
model.save('toda_okura_inceptionv3_test.h5')

# Save training parameters
print('Obtaining training data...')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Print data
print_training_data(acc, val_acc, loss, val_loss)
write_data_to_file(acc, val_acc, loss, val_loss, 'data_color.csv')
plot_graphs(acc, val_acc, loss, val_loss)

# Get testing results 
test_results = model.evaluate(test_x, test_y)
print("Printing test results...")
for metric, name, in zip(test_results, ['loss', 'acc', 'top 5 acc']):
  print(name, metric)
print("Model finished!")
