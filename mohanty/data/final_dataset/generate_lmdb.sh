#!/bin/bash

#SBATCH --workdir /scratch/user/skumar55/mohanty/data/final_dataset
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 16384
#SBATCH --time 23:59:59
#SBATCH --partition gpu
#SBATCH --gres gpu:2
##SBATCH --qos gpu

#SBATCH --account=122749846190

module load Anaconda/3-5.0.0.1
source activate caffe-gpu-1.0
source activate caffe_gpu
echo STARTING AT `date`


for _config in `ls lmdb`
do
	echo lmdb/$_config
	cp _generate_data.sh generate_data_$_config.sh
	echo "python create_db.py -b lmdb -s -r squash -c 3 -e jpg -C gzip -m lmdb/$_config/mean.binaryproto  lmdb/$_config/train.txt lmdb/$_config/train_db 256 256" >> generate_data_$_config.sh
	echo "python create_db.py -b lmdb -s -r squash -c 3 -e jpg -C gzip  lmdb/$_config/test.txt lmdb/$_config/test_db 256 256" >> generate_data_$_config.sh
done
