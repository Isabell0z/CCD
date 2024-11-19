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
│     └─teachers (teacher models saved here)
│        └─MFSelf
│          └─TASK_0.pth (example)
├─ensemble_utils
│  └─ensemble.py
├─KD_utils
│  ├─dataset.py
│  ├─DE.py
│  ├─kd.py
│  ├─RRD.py
│  ├─save_mat.py
│  └─utils.py
├─self_models
│  ├─BaseModels.py
│  └─LWCKD.py
├─run_scripts
│  ├─run.sh
│  └─runTransformer.sh
├─Utils
│  ├─data_loader.py
│  ├─eval.py
│  ├─fix_scoremat.py
│  ├─load_model.py
│  └─utils.py
├─kd_.py
├─student_update.py
├─teacher_update.py
├─train_models.py
└─README.md
```

# Run
You can test each of the below commands and check whether they can run successfully.
Or you can run `bash run_scripts/run.sh` directly which will train the teacher model and process 3 rounds of CCD.
## Train Teacher Models

Can be skipped if models are prepared.

`python train_models.py --dataset {data} --model {model type} --cuda {device id} --max_epoch 1000`

- data: Yelp or Gowalla
- model type: MFSelf or TransformerSelf
- device id: default -> 0

## Knowledge Distillation

`python kd.py --dataset {data} --model {model type} --tt {tt} --max_epoch 10`

- data: Yelp or Gowalla
- model type: MFSelf or TransformerSelf
- tt: target task

## Student Update

`python student_update.py --d Yelp -m MFSelf --tt 1 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 10 --s`

## Teacher Update

`python teacher_update.py --d Yelp --student MFSelf --teacher MFSelf --tt 1 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 10 --s`

