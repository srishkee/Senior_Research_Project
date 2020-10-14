from keras.models import load_model
import os
import numpy as np
import random
import glob
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import np_utils
# from keras.preprocessing.image import ImageDataGenerator

# This evaluates the InceptionV3 model on the test dataset.

model = load_model('toda_okura_inceptionv3.h5')
model.summary()


# plant_color = 'color' # 3 categories: color/grayscale/segmented
# test_dir = 'plant_data_small_3/'+plant_color+'/test'
# test_datagen = ImageDataGenerator(rescale=1./255) # Don't augment test data!
# test_generator = test_datagen.flow_from_directory(directory=test_dir, target_size=(224,224), batch_size=128, class_mode='categorical')

def load_plantvillage(seed=None, root_dir=None):
  if(root_dir==None):
    print("Please enter a valid data directory!")
    return
  
  random.seed(seed)

  def read_from_paths(paths):
    x=[]
    for path in paths:
      img = load_img(path, target_size=(224,224))
      img = img_to_array(img)
      x.append(img)
    return x

  classes = os.listdir(root_dir)
  classes = sorted(classes)

  test_path = []
  test_x, test_y = [],[]

  # Read paths and split data into 6:2:2 dataset
  for i, _class in enumerate(classes):
    paths = glob.glob(os.path.join(root_dir, _class, "*"))
    paths = [n for n in paths if n.endswith(".JPG") or n.endswith(".jpg")]
    random.shuffle(paths)
    num_plants = len(paths)
    test_path.extend(paths[int(num_plants*0.8):])
    test_y.extend([i]*len(paths[int(num_plants*0.8):]))

  print("Loading images...")
  test_x = read_from_paths(test_path)
  test_x = np.array(test_x)/255.
  print("Loaded all testing images!")

  test_y = np.array(test_y)
  test_y = np_utils.to_categorical(test_y, 38)
  print("Successfully loaded data!")

  return test_x, test_y

# ------------------------------ Main ------------------------------ #
# This part will take a while
data_dir = '/scratch/user/skumar55/plantvillage_deeplearning_paper_dataset/raw/color/'
test_x, test_y = load_plantvillage(seed=7, root_dir=data_dir)

print(test_x.shape)   
print(test_y.shape)

# Evaluate model on testing data
test_results = model.evaluate(test_x, test_y)
print("Printing test results...")
for metric, name, in zip(test_results, ['loss', 'acc', 'top 5 acc']):
  print(name, metric)
print("Model finished!")


# print('Testing model on test data...')
# test_loss, test_acc = model.evaluate(x=test_generator, verbose=1) # Add 'steps' parameter? 
# print('test_loss: ', test_loss)
# print('test_acc: ', test_acc)