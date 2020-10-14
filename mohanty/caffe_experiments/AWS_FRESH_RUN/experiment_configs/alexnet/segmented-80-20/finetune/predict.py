import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import caffe

# This function gets the predicted class for a given image
def get_predictions(MODEL_FILE, PRETRAINED, IMG_PATH):
	# load the model
	#caffe.set_mode_gpu()
	#caffe.set_device(0)
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
	                       mean=np.load('train_mean.npy').mean(1).mean(1),
	                       channel_swap=(2,1,0),
	                       raw_scale=255,
	                       image_dims=(256, 256))

	print("Successfully loaded classifier!")

	# Declare image path
	IMAGE_FILE = IMG_PATH
	# Load image
	input_image = caffe.io.load_image(IMAGE_FILE)
	# Get predictions. Note: predict takes any number of images, and formats them for the Caffe net automatically
	predictions = net.predict([input_image])
	return predictions

# This function gets max prediction parameters
def get_max_predictions(predictions):
	# Get max prediction (and its index)
	max_pred = np.max(predictions)
	# Get index of max prediction 
	max_pred_idx = np.argmax(predictions)

	# Parse list of labels into array
	filename = 'labels.txt'
	pred_labels = []
	with open(filename) as f:
		pred_labels = f.readlines()

	# Get predicted label
	max_label = pred_labels[max_pred_idx]
	max_label = max_label[:-1] # Remove \n\r characters 

	# Return max predictions & label
	return (max_pred, max_pred_idx, max_label)

# This function gets the expected label mappings, as specified in 'label_mappings.txt'
def get_expected_labels():

	expected_labels = []

	filename = 'label_mappings.txt'
	with open(filename) as f:
		expected_labels = f.readlines()
	return expected_labels


# This function prints the output to the console
def pretty_print(FOLDER, PLANT_COLOR, IMG_IDX, imgs, max_pred_idx, max_label, max_pred, predictions):
	# Print final answer 
	print("\n\n")
	print("------------------------------------------------------------------------------------")
	print("\nSuccessfully classified image!\n")
	input = "INPUT: Image '" + imgs[IMG_IDX] + "'\n"
	print(input)
	print("PLANT TYPE: " + PLANT_COLOR + "\n")
	expected_labels = get_expected_labels()
	print("EXPECTED: " + expected_labels[FOLDER])
	ans = "PREDICTION: Category " + str(max_pred_idx) + ", '" + max_label +  "' with an accuracy = " + str(max_pred*100) + "%\n"
	print(ans)
	# predictions = [x*100 for x in predictions]
	# print(predictions) # Predictions are in order of the PlantVillage Github!
	print("------------------------------------------------------------------------------------")
	print("\n\n")


# -------------------------- main -------------------------- #

# Set the right path to your model definition file, pretrained model weights, and the image you would like to classify.
MODEL_FILE = 'deploy.prototxt'
PRETRAINED = '/scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_segmented-80-20_finetune/snapshots__iter_13057.caffemodel'
PLANT_COLOR = 'segmented' # Options: color/grayscale/segmented
FOLDER = 37
IMG_IDX = 55
IMG_DIR = '/scratch/user/skumar55/plant_data_small_3/' + PLANT_COLOR + '/test/' + str(FOLDER) + '/'

# Configure image parameters 
imgs = os.listdir(IMG_DIR)
IMG_PATH = IMG_DIR + imgs[IMG_IDX]

# -------------------------- predict -------------------------- #

predictions = get_predictions(MODEL_FILE, PRETRAINED, IMG_PATH)

(max_pred, max_pred_idx, max_label) = get_max_predictions(predictions)

pretty_print(FOLDER, PLANT_COLOR, IMG_IDX, imgs, max_pred_idx, max_label, max_pred, predictions)


