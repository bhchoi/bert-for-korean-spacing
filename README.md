# bert-for-korean-spacing
Pretrained BERT를 이용한 한국어 띄어쓰기

## Dataset

## Train
* train_config.yaml
```yaml
log_path: logs
bert_model: beomi/kcbert-base
train_data_path: data/kcbert/train_data.txt
val_data_path: data/kcbert/val_data.txt
test_data_path: data/kcbert/test_data.txt
max_len: 128
train_batch_size: 64
eval_batch_size: 32
dropout_rate: 0.1
gpus: 1
distributed_backend: ddp
```

```python
python train.py
```

## Eval
* eval_config.yaml
```yaml
bert_model: beomi/kcbert-base
test_data_path: data/kcbert/test_data.txt
chpt_path: checkpoints/epoch=4_val_acc=0.000000.ckpt
max_len: 128
eval_batch_size: 32
dropout_rate: 0.1
```

```python
python eval.py
```

## Reference
* https://github.com/Beomi/KcBERT
