# Import necessary modules
import os
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from models import AttentionResNetCifar10
from models import AttentionResNet56

# Define parameters
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NAME = 'AttentionResNet56'

# print("Running AttentionResNet56 on Cifar10 data!")
print("CHANGE: Running ", NAME)

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

# ------------------------------ Main ------------------------------ #
# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# define generators for training and validation data
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
train_datagen.fit(x_train)
val_datagen.fit(x_train)

# build a model
# model = AttentionResNetCifar10(n_classes=10)
model = AttentionResNet56(shape=(32, 32, 3), n_channels=32, n_classes=10)

# prepare usefull callbacks
lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=7, min_lr=10e-7, epsilon=0.01, verbose=1)
early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=1)
model_checkpoint = ModelCheckpoint(filepath='saved_model_weights/'+NAME+'.h5', monitor='val_loss', verbose=1, save_weights_only=False, period=1)
callbacks= [lr_reducer, early_stopper, model_checkpoint]

# define loss, metrics, optimizer
model.compile(keras.optimizers.Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# fits the model on batches with real-time data augmentation
history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(x_train)//BATCH_SIZE, epochs=EPOCHS,
                    validation_data=val_datagen.flow(x_test, y_test, batch_size=BATCH_SIZE), 
                    validation_steps=len(x_test)//BATCH_SIZE,
                    callbacks=callbacks, initial_epoch=0)
print ("Successfully fit model to data!")

model.save(NAME + '.h5')
print("Saved model successfully!")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Print data
print_training_data(acc, val_acc, loss, val_loss)
write_data_to_file(acc, val_acc, loss, val_loss, NAME+'.csv')
plot_graphs(acc, val_acc, loss, val_loss)

model.evaluate_generator(val_datagen.flow(x_test, y_test), steps=len(x_test)/BATCH_SIZE, use_multiprocessing=True)
print ("Evaluated model!")


