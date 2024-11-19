#!/bin/sh
python train_models.py --dataset Yelp --model MFSelf --max_epoch 1000
wait
python kd_.py --dataset Yelp --model MFSelf --tt 0 --max_epoch 10 --s
python student_update.py --d Yelp -m MFSelf --tt 1 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 10 --s
python teacher_update.py --d Yelp --student MFSelf --teacher MFSelf --tt 1 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 10 --s
wait
python kd_.py --dataset Yelp --model MFSelf --tt 1 --max_epoch 10 --s
python student_update.py --d Yelp -m MFSelf --tt 2 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 10 --s
python teacher_update.py --d Yelp --student MFSelf --teacher MFSelf --tt 2 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 10 --s
wait
python kd_.py --dataset Yelp --model MFSelf --tt 2 --max_epoch 10 --s
python student_update.py --d Yelp -m MFSelf --tt 3 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 10 --s
python teacher_update.py --d Yelp --student MFSelf --teacher MFSelf --tt 3 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 10 --s
wait
python kd_.py --d Yelp -m MFSelf --tt 3 --max_epoch 10 --s

# date
# squeue --job $SLURM_JOBID
# echo "##### END ####"
