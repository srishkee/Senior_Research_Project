#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Googlenet_Grayscale_80_20_Test       #Set the job name to "JobExample2"
#SBATCH --workdir /scratch/user/skumar55/mohanty/caffe_experiments/AWS_FRESH_RUN/experiment_configs/googLeNet/grayscale-80-20/finetune
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 16384
#SBATCH --time 23:59:59
#SBATCH --partition gpu
#SBATCH --gres gpu:2
##SBATCH --qos gpu
#SBATCH --output=test_80_20.%j      #Send stdout/err to "Example2Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=122749846190         #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on 0 job events
#SBATCH --mail-user=skumar55@tamu.edu      #Send 0 emails to email_address

module load Anaconda/3-5.0.0.1
source activate caffe-gpu-1.0
source activate caffe_gpu
echo STARTING AT `date`

export PATH=/scratch/user/skumar55/mohanty/caffe/build/inst0/bin/:$PATH


time caffe test -model test_prototxts/iter_1817.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_1817.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_1817.log
time caffe test -model test_prototxts/iter_3634.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_3634.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_3634.log
time caffe test -model test_prototxts/iter_5451.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_5451.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_5451.log
time caffe test -model test_prototxts/iter_7268.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_7268.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_7268.log
time caffe test -model test_prototxts/iter_9085.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_9085.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_9085.log
time caffe test -model test_prototxts/iter_10902.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_10902.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_10902.log
time caffe test -model test_prototxts/iter_12719.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_12719.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_12719.log
time caffe test -model test_prototxts/iter_14536.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_14536.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_14536.log
time caffe test -model test_prototxts/iter_16353.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_16353.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_16353.log
time caffe test -model test_prototxts/iter_18170.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_18170.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_18170.log
time caffe test -model test_prototxts/iter_19987.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_19987.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_19987.log
time caffe test -model test_prototxts/iter_21804.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_21804.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_21804.log
time caffe test -model test_prototxts/iter_23621.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_23621.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_23621.log
time caffe test -model test_prototxts/iter_25438.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_25438.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_25438.log
time caffe test -model test_prototxts/iter_27255.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_27255.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_27255.log
time caffe test -model test_prototxts/iter_29072.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_29072.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_29072.log
time caffe test -model test_prototxts/iter_30889.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_30889.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_30889.log
time caffe test -model test_prototxts/iter_32706.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_32706.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_32706.log
time caffe test -model test_prototxts/iter_34523.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_34523.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_34523.log
time caffe test -model test_prototxts/iter_36340.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_36340.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_36340.log
time caffe test -model test_prototxts/iter_38157.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_38157.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_38157.log
time caffe test -model test_prototxts/iter_39974.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_39974.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_39974.log
time caffe test -model test_prototxts/iter_41791.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_41791.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_41791.log
time caffe test -model test_prototxts/iter_43608.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_43608.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_43608.log
time caffe test -model test_prototxts/iter_45425.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_45425.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_45425.log
time caffe test -model test_prototxts/iter_47242.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_47242.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_47242.log
time caffe test -model test_prototxts/iter_49059.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_49059.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_49059.log
time caffe test -model test_prototxts/iter_50876.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_50876.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_50876.log
time caffe test -model test_prototxts/iter_52693.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_52693.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_52693.log
time caffe test -model test_prototxts/iter_54510.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_54510.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_54510.log
time caffe test -model test_prototxts/iter_54522.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/googLeNet_grayscale-80-20_finetune/snapshots__iter_54522.caffemodel -gpu 0 -iterations 446 &> ./test_logs/iter_56327.log

#sbatch results/generate_results.sh

echo FINISHED at `date`

