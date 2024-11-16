#!/bin/sh
python kd.py --dataset Yelp --model TransformerSelf --tt 0 --max_epoch 100 
python student_update.py --d Yelp -m TransformerSelf --tt 1 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s
python teacher_update.py --d Yelp --student TransformerSelf --teacher TransformerSelf --tt 1 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s
wait
python kd.py --dataset Yelp --model TransformerSelf --tt 1 --max_epoch 10  
python student_update.py --d Yelp -m TransformerSelf --tt 2 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s
python teacher_update.py --d Yelp --student TransformerSelf --teacher TransformerSelf --tt 2 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s
wait
python kd.py --dataset Yelp --model TransformerSelf --tt 2 --max_epoch 10  
python student_update.py --d Yelp -m TransformerSelf --tt 3 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s
python teacher_update.py --d Yelp --student TransformerSelf --teacher TransformerSelf --tt 3 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s
wait
python kd.py --dataset Yelp --model TransformerSelf --tt 3 --max_epoch 10  
python student_update.py --d Yelp -m TransformerSelf --tt 4 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s
python teacher_update.py --d Yelp --student TransformerSelf --teacher TransformerSelf --tt 4 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s
wait
python kd.py --dataset Yelp --model TransformerSelf --tt 4 --max_epoch 10  
python student_update.py --d Yelp -m TransformerSelf --tt 5 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s
python teacher_update.py --d Yelp --student TransformerSelf --teacher TransformerSelf --tt 5 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s
wait
python KD.py --d Yelp -m TransformerSelf --tt 5 --max_epoch 10

# date
# squeue --job $SLURM_JOBID
# echo "##### END ####"
