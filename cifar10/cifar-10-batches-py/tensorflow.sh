#!/bin/bash -l
#$ -pe omp 2
#$ -N cifar_test
#$ -l gpus=2
#$ -l gpu_type=K40m
module load python/2.7.12
module load cuda/8.0
module load cudnn/5.1
module load tensorflow/r0.12

python2 mtezcan_CNN_cifar.py
