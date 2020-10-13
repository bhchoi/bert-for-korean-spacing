## bert-for-korean-spacing
BERT Pretrained model을 이용한 한국어 띄어쓰기

## Dataset
* 

## Train
* train_config.yaml
```yaml
log_path: logs
bert_model: monologg/kobert
train_data_path: data/train_data.txt
val_data_path: data/val_data.txt
test_data_path: data/test_data.txt
max_len: 128
train_batch_size: 64
eval_batch_size: 64
dropout_rate: 0.1
gpus: 8
distributed_backend: ddp
```

```python
python train.py
```

## Eval
* eval_config.yaml
```yaml
bert_model: monologg/kobert
test_data_path: data/test_data.txt
ckpt_path: checkpoints/epoch=4_val_acc=0.000000.ckpt
max_len: 128
eval_batch_size: 64
dropout_rate: 0.1
```

```python
python eval.py
```
## Results
* 

## Example
* 

## Reference
* https://github.com/monologg/KoBERT-Transformers
* https://github.com/PyTorchLightning/pytorch-lightning
* https://github.com/omry/omegaconf
