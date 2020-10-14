import os

# This script parses the test iteration files for the loss and accuracy values. 

def get_files(PATH):
	# Get all filenames in directory (sorted by time) 
	files = (os.listdir(PATH))
	files.sort(key=lambda x: os.path.getmtime(PATH+x))
	return files


def parse_files(PATH, target_str):

	files = get_files(PATH) # Get files
	values = []
	for filename in files: # Iterate over all files
		
		content = '' # Read file
		with open(PATH+filename, 'r') as f:
			content = f.read()
		f.close()

		# Get loss/accuracy value
		start_pos = content.find(target_str) + len(target_str) # Get starting position
		end_pos = content.find('\n', start_pos) # Get ending position
		val = content[start_pos:end_pos] # Get loss/accuracy value
		if(len(val) < 10): 
			# print(filename)
			# print(val)
			values.append(val)

	return values


def write_to_file(PATH, filename, values):
	open(filename, 'w+').close() # Clear file before writing to it (and create if nonexistent)
	with open(filename, 'a') as output_file: # Open file 
		for val in values: 
			output_file.write(val + '\n') # Write all loss/accuracy values
	output_file.close()

# -------------------------- main -------------------------- #

# Define path to iter_ files
PATH = '/scratch/user/skumar55/mohanty/caffe_experiments/AWS_FRESH_RUN/experiment_configs/alexnet/color-80-20/finetune/test_logs/'

# Define target string (final loss value occurs immediately afterwards)
target_str_acc = 'caffe.cpp:330] accuracy = '
target_str_loss = 'caffe.cpp:318] Loss: '

# Note: these files are created within the current directory (finetune)
acc_file = 'test_acc_values.txt'
loss_file = 'test_loss_values.txt'

files = get_files(PATH)
acc_values = parse_files(PATH, target_str_acc) 
loss_values = parse_files(PATH, target_str_loss)

write_to_file(PATH, acc_file, acc_values)
print("Successfully parsed accuracy values!")
write_to_file(PATH, loss_file, loss_values)
print("Successfully parsed loss values!")

