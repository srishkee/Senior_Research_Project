This folder contains the code for the InceptionV3 model. 

**Running Code:** Run `MyJob2.slurm` (bash file, so run on Terra if using Tamu HPRC) with the appropriate file name to run a script. 

Eg: To train the CNN model, simply execute `sbatch MyJob2.slurm` on Terra. 

`feature_visualization/:` Contains a Jupyter script that performs visualization of desired CNN channels. 

`plots/:` Contains Training/Validation accuracy/loss plots. Generated while training the model. 

`data.csv:` Contains the Training/Validation accuracy/loss raw data. 

`final_inception_v3.py:` My initial attempt to write a working model (not entirely correct).

`generate_plant_distribution:` Script to generate the 60:20:20 data distribution for the `final_inception_v3.py` script.

`testing_inception_v3.py:` Script to test the trained CNN model on the testing data. 

`toda_okura_inceptionv3.py:` Python script that correctly builds the CNN model (original author's code). 

`toda_okura_resnet50.py:` Python script to train ResNet50 on PlantVillage data, based off of the author's implementation. 

**Note:** The `.h5` files containing the trained model weights are not included in this repo due to their size. 