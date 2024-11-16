# CCD
data mining

# Directory

```
├─dataset
│  ├─Gowalla
│  └─Yelp
├─ckpts
│  ├─Gowalla
│  └─Yelp
│     ├─students (student models saved here)
│     │  
│     └─teachers (teacher models put here)
│        ├─MFSelf
│        │ └─TASK_0.pth
│        ├─TransformerSelf
│        │ └─TASK_0.pth
│        └─VAESelf
│          └─TASK_0.pth
├─ensemble_utils
├─KD_utils
├─llw
│  ├─dataset
│  │  ├─Gowalla
│  │  └─Gowalla_new0
│  ├─data_process
│  └─Utils
├─originalCCD
│  ├─figure
│  ├─Models
│  ├─Run
│  └─Utils
├─self_models
└─Utils
```

# Run
## Train Teacher Models

Can be skipped if models are prepared.

`python train_models.py --dataset {data} --model {model type} --cuda {device id}`

- data: Yelp or Gowalla
- model type: TransformerSelf, VAESelf or MFSelf
- device id: default -> 0

## Knowledge Distillation

`python kd.py --dataset {data} --model {TransformerSelf} --tt {tt} --max_epoch 10`

- data: Yelp or Gowalla
- model type: TransformerSelf, VAESelf or MFSelf
- device id: default -> 0
- tt: target task

## Student Update

`python student_update.py --d Yelp -m TransformerSelf --tt 1 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 50 --s`

## Teacher Update

`python -u teacher_update.py --d Yelp --student TransformerSelf --teacher TransformerSelf --tt 1 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 50 --s`

