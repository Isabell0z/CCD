#!/bin/sh
python kd_.py --dataset Yelp --model MFSelf --tt 0 --max_epoch 100 --s
python s_update.py --d Yelp -m MFSelf --tt 1 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s
python t_update.py --d Yelp --student MFSelf --teacher MFSelf --tt 1 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s
wait
python kd_.py --dataset Yelp --model MFSelf --tt 1 --max_epoch 100 --s
python s_update.py --d Yelp -m MFSelf --tt 2 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s
python t_update.py --d Yelp --student MFSelf --teacher MFSelf --tt 2 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s
wait
python kd_.py --dataset Yelp --model MFSelf --tt 2 --max_epoch 100 --s
python s_update.py --d Yelp -m MFSelf --tt 3 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s
python t_update.py --d Yelp --student MFSelf --teacher MFSelf --tt 3 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s
wait
# python kd.py --dataset Yelp --model MFSelf --tt 3 --max_epoch 50  
# python s_update.py --d Yelp -m MFSelf --tt 4 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s
# python s_update.py --d Yelp --student MFSelf --teacher MFSelf --tt 4 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s
# wait
# python kd.py --dataset Yelp --model MFSelf --tt 4 --max_epoch 50  
# python s_update.py --d Yelp -m MFSelf --tt 5 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s
# python s_update.py --d Yelp --student MFSelf --teacher MFSelf --tt 5 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s
# wait
python kd_.py --d Yelp -m MFSelf --tt 3 --max_epoch 100 --s

# date
# squeue --job $SLURM_JOBID
# echo "##### END ####"
