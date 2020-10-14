#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Alex_Grayscale_80_20_Test       #Set the job name to "JobExample2"
#SBATCH --workdir /scratch/user/skumar55/mohanty/caffe_experiments/AWS_FRESH_RUN/experiment_configs/alexnet/grayscale-80-20/finetune
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

time caffe test -model test_prototxts/iter_436.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_436.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_436.log
time caffe test -model test_prototxts/iter_872.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_872.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_872.log
time caffe test -model test_prototxts/iter_1308.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_1308.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_1308.log
time caffe test -model test_prototxts/iter_1744.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_1744.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_1744.log
time caffe test -model test_prototxts/iter_2180.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_2180.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_2180.log
time caffe test -model test_prototxts/iter_2616.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_2616.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_2616.log
time caffe test -model test_prototxts/iter_3052.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_3052.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_3052.log
time caffe test -model test_prototxts/iter_3488.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_3488.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_3488.log
time caffe test -model test_prototxts/iter_3924.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_3924.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_3924.log
time caffe test -model test_prototxts/iter_4360.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_4360.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_4360.log
time caffe test -model test_prototxts/iter_4796.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_4796.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_4796.log
time caffe test -model test_prototxts/iter_5232.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_5232.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_5232.log
time caffe test -model test_prototxts/iter_5668.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_5668.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_5668.log
time caffe test -model test_prototxts/iter_6104.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_6104.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_6104.log
time caffe test -model test_prototxts/iter_6540.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_6540.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_6540.log
time caffe test -model test_prototxts/iter_6976.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_6976.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_6976.log
time caffe test -model test_prototxts/iter_7412.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_7412.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_7412.log
time caffe test -model test_prototxts/iter_7848.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_7848.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_7848.log
time caffe test -model test_prototxts/iter_8284.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_8284.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_8284.log
time caffe test -model test_prototxts/iter_8720.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_8720.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_8720.log
time caffe test -model test_prototxts/iter_9156.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_9156.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_9156.log
time caffe test -model test_prototxts/iter_9592.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_9592.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_9592.log
time caffe test -model test_prototxts/iter_10028.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_10028.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_10028.log
time caffe test -model test_prototxts/iter_10464.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_10464.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_10464.log
time caffe test -model test_prototxts/iter_10900.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_10900.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_10900.log
time caffe test -model test_prototxts/iter_11336.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_11336.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_11336.log
time caffe test -model test_prototxts/iter_11772.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_11772.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_11772.log
time caffe test -model test_prototxts/iter_12208.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_12208.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_12208.log
time caffe test -model test_prototxts/iter_12644.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_12644.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_12644.log
time caffe test -model test_prototxts/iter_13080.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_13080.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_13080.log
time caffe test -model test_prototxts/iter_13085.prototxt -weights /scratch/user/skumar55/mohanty/AWS_FRESH_RUN/snapshots_final/alexnet_grayscale-80-20_finetune/snapshots__iter_13085.caffemodel -gpu 0 -iterations 107 &> ./test_logs/iter_13516.log

#sbatch results/generate_results.sh

echo FINISHED at `date`
