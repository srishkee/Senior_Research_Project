#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=google_80_20       #Set the job name to "JobExample2"
#SBATCH --workdir /scratch/user/skumar55/mohanty/caffe_experiments/AWS_FRESH_RUN/experiment_configs/googLeNet/color-80-20/finetune
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 16384
#SBATCH --time 23:59:59
#SBATCH --partition gpu
#SBATCH --gres gpu:2
##SBATCH --qos gpu
#SBATCH --output=UCFcountingTestOut.%j      #Send stdout/err to "Example2Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=122749846190         #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=skumar55@tamu.edu      #Send all emails to email_address

module load Anaconda/3-5.0.0.1
source activate caffe-gpu-1.0
source activate caffe_gpu
echo STARTING AT `date`

caffe train -solver solver.prototxt -weights /scratch/user/skumar55/mohanty/caffe_experiments/AWS_FRESH_RUN/models/bvlc_googlenet.caffemodel -gpu 0 &> caffe.log
#sbatch test.sh
echo FINISHED at `date`

