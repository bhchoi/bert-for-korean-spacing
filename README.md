## bert-for-korean-spacing
BERT Pretrained model을 이용한 한국어 띄어쓰기

## Dataset
* 세종코퍼스 1,037,330 문장

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
* testset : 103,733건
* SER : 0.277
* F1 score : 0.966 

## Example
>

## Reference
* https://github.com/monologg/KoBERT-Transformers
* https://github.com/PyTorchLightning/pytorch-lightning
* https://github.com/omry/omegaconf
