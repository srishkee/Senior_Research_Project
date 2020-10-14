This folder contains scripts to create the LMDB dataset for the AlexNet and GoogLeNet models. 

An LMDB (Lightening Memory-Mapped Database) dataset is preferrable for these models since it allows faster IO access. 

This is a rather complicated folder that you shouldn't be too worried about. Only a few commands should be necessary to create the database. 

`lmdb/:` Contains the LMDB dataset, organized by category. 

**Creating the dataset:** To create an lmdb for a certain plant category (eg. `color-80-20`), simply run the corresponding script (eg. `generate_data_color-80-20.sh`).  

Alternatively, you can run `run_all.sh` to create the lmdb for all categories; however, I found that was very time-consuming, and used the above method instead. 

After creating the dataset, you will find the folders `train.db` and `test.db` (not included in this repo due to their size) in their respective category within the `lmdb/` folder. 
Each will contain a `lock.mdb` and `data.mdb` file. 

**Misc:** The remainder of the files/folders (`utils/`, `.py`, `.sh`) can be ignored, unless they need to be updated to a newer version. 
In that case, you can try fixing errors with the Python conversion tool `python 2to3`, or manually resolve the errors. 