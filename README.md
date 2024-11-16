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
│        ├─MFSelf_TASK_0.pth
│        ├─TransformerSelf_TASK_0.pth
│        └─VAESelf_TASK_0.pth
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

## Knowledge Distillation

`python kd.py --dataset {data} --model {model type} --cuda {device id}`

- data: Yelp or Gowalla
- model type: TransformerSelf, VAESelf or MFSelf
- device id: default -> 0

## Student Update

`python -u student_update.py --d Yelp -m TransformerSelf_TASK_0 --tt 1 --rl --US --UP --ab 50 --ss 1 --ps 0 --sw 1.0 --pw 0.0 --max_epoch 10`

## Teacher Update

todo
