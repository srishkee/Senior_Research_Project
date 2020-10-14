import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
import random
import glob
from sklearn.utils import class_weight

import keras
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import np_utils
from keras.optimizers import SGD, adam
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import optimizers

from models import attention_block

# Define parameters
NAME = 'ResNet50_experiment'
backend = None
models = None
keras_utils = None
EPOCHS = 50
LEARNING_RATE = 0.005
BATCH_SIZE = 32

print("\nCHANGE: Running SIMPLIFIED ResNet50_experiment! Only uses 2 attention blocks, and removed 3rd conv segment. encoder_depths = {2,1}\n")

data_dir = '/scratch/user/skumar55/plantvillage_deeplearning_paper_dataset/raw/color/'

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
    data.insert(1,"Validation Acc", val_acc)
    data.insert(2,"Training Loss", loss)
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

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    # if backend.image_data_format() == 'channels_last':
    bn_axis = 3
    # else:
    #     bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    # if backend.image_data_format() == 'channels_last':
    bn_axis = 3
    # else:
        # bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = keras.layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x

def get_resnet50(input_shape=(224,224,3), num_classes=38):


    # global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    bn_axis = 3

    img_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = keras.layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = attention_block(x, encoder_depth=2)
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = attention_block(x, encoder_depth=1)
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    # x = attention_block(x, encoder_depth=1)

    # Changed filter & stage values to match Stage#4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')

    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax', name='prediction')(x)
    model = Model(inputs=img_input, outputs=predictions, name='resnet50')
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=LEARNING_RATE), metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model

# ------------------------------ Main ------------------------------ #
# Get model
model = get_resnet50()
print("Successfully loaded model!")

# This part will take a while
train_x, val_x, test_x, train_y, val_y, test_y, classweight, classes = load_plantvillage(seed=7, root_dir=data_dir)

print(train_x.shape, val_x.shape, test_x.shape)   
print(train_y.shape, val_y.shape, test_y.shape)

es = EarlyStopping(patience=15)
history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_x, val_y), callbacks=[es], class_weight=classweight)

# Save model 
print ("Saving model...")
model.save(NAME + '.h5')

# Save training parameters
print('Obtaining training data...')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Print data
print_training_data(acc, val_acc, loss, val_loss)
write_data_to_file(acc, val_acc, loss, val_loss, NAME+'.csv')
plot_graphs(acc, val_acc, loss, val_loss)

# Get testing results 
test_results = model.evaluate(test_x, test_y)
print("Printing test results...")
for metric, name, in zip(test_results, ['loss', 'acc', 'top 5 acc']):
  print(name, metric)
print("Model finished!")
