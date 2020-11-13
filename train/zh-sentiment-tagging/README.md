### 


### argparse sample case

```json
{"output_dir": "test", 
 "fp16": False, 
 "fp16_opt_level": "O2", 
 "tpu_cores": None, 
 "gradient_clip_val": 1.0, 
 "do_train": False, 
 "do_predict": False, 
 "accumulate_grad_batches": 1, 
 "seed": 42, 
 "data_dir": "data/step-00/basic", 
 "model_name_or_path": "hfl/chinese-electra-180g-large-discriminator", 
 "config_name": "", 
 "tokenizer_name": None, 
 "cache_dir": "", 
 "encoder_layerdrop": None, 
 "decoder_layerdrop": None, 
 "dropout": None, 
 "attention_dropout": None, 
 "learning_rate": 5e-05, 
 "lr_scheduler": "linear", 
 "weight_decay": 0.0, 
 "adam_epsilon": 1e-08, 
 "warmup_steps": 0, 
 "num_workers": 4, 
 "max_epochs": 3,  
 "train_batch_size": 32, 
 "eval_batch_size": 32, 
 "adafactor": False, 
 "task_type": "SentimentTagging", 
 "max_seq_length": 128, 
 "labels": "", 
 "gpus": 0, 
 "overwrite_cache": False}
```