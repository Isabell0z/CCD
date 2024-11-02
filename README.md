# CCD
data mining

run: (task 2-5 run after KD)


task 1:

python -u Student_update.py --d Yelp -m LightGCN_1 --tt 1 --rl --US --UP --ab 50 --ss 1 --ps 0 --sw 1.0 --pw 0.0 --max_epoch 10

task 2:

python -u Student_update.py --d Yelp -m LightGCN_1 --tt 2 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 10

task 3:

python -u Student_update.py --d Yelp -m LightGCN_1 --tt 3 --rl --US --UP --ab 100 --ss 5 --ps 5 --sw 0.9 --pw 0.0 --max_epoch 10

task 4:

python -u Student_update.py --d Yelp -m LightGCN_1 --tt 4 --rl --US --UP --ab 50 --ss 1 --ps 5 --sw 0.9 --pw 0.0  --max_epoch 10

task 5:

python -u Student_update.py --d Yelp -m LightGCN_1 --tt 5 --rl --US --UP --ab 50 --ss 3 --ps 1 --sw 0.9 --pw 0.0 --max_epoch 10


student updata:

1. use original utils.py

2. evaluation part: 

a)student, P and S proxy after KD

b)best student model
