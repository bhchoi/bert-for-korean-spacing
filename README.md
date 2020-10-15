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
> input  : 그냥영풍이라고써있으니까될거같지않냐?  
> output : 그냥 영풍이라고 써 있으니까 될 거 같지 않냐?

> input  : 대표적인미디어문화연구자인더글러스켈너는이렇게말하고있다.  
> output : 대표적인 미디어 문화연구자인 더글러스 켈너는 이렇게 말하고 있다.

> input  : 트렁크룸사업의성장성은이례적이다.	  
> output : 트렁크 룸사업의 성장성은 이례적이다.

## Reference
* https://github.com/monologg/KoBERT-Transformers
* https://github.com/PyTorchLightning/pytorch-lightning
* https://github.com/omry/omegaconf
