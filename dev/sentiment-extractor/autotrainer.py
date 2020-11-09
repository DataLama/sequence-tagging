import os
import sys
import json
import logging
import shutil
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, field
from tqdm.auto import tqdm, trange
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # test
import numpy as np
import pandas as pd
import nni
from torch.utils.tensorboard import SummaryWriter

from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed, EvalPrediction
from models.data import SentimentExtractionDataset, SentimentExtractionDataTrainingArguments, Split, hf_efficient_data_collator
from models.utils import (
    ModelArguments, 
    MODEL_PATH_MAP,
    CONFIG_CLASSES, 
    TOKENIZER_CLASSES, 
    MODEL_FOR_TOKEN_CLASSIFICATION,
    init_logger,
    label_map,
    compute_metrics
)

from transformers.convert_graph_to_onnx import convert

logger = logging.getLogger(__name__)
HISTORY = list()

## overwriting _log add nni
class NNITrainer(Trainer):
    """DataLama Custom Trainer
    Base Code - hugginface transformers 3.0.2
    
    eradicate wandb
    """
    
    history = [] # get train log
    
    
    def _log(self, logs, iterator=None):
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        if self.tb_writer:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self.tb_writer.flush()
        output = {**logs, **{"step": self.global_step}}
        #### nni
        if (nni is not None) and ('eval_token-f1' in logs):
            nni.report_intermediate_result(logs['eval_token-f1'])
        ####
        if 'eval_loss' in output.keys():
            self.history.append(output)
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(output)
            
    def get_best_model(self, metric, app_key, app_name=None):
        """Get best model and scripting the model, (metric='token-f1')"""

        # Get best model by validation metrics
        eval_history = self.history

        if metric in {'token-f1'}:
            bestscore = np.max([m[f'eval_{metric}'] for m in eval_history])
            best_step = eval_history[np.argmax([m[f'eval_{metric}'] for m in eval_history])]['step']
        elif metric in {'loss'}:
            bestscore = np.min([m[f'eval_{metric}'] for m in eval_history])
            best_step = eval_history[np.argmin([m[f'eval_{metric}'] for m in eval_history])]['step']
        else:
            raise ValueError(f'Use Another metric {metric} is not supported.')
                
        best_dir = os.path.join(self.args.output_dir, f'checkpoint-{best_step}')
        print(f'my best model {metric} is {bestscore} at step {best_step}')
        
        # onnx export
        train_time = app_key.split('/')[0]
        target_dir = f"{app_name}/{app_key},{metric}={bestscore:.4f}/SentimentExtractor-{train_time}"
        os.makedirs(target_dir, exist_ok=True)
        
        convert(
            framework='pt', 
            model=best_dir, 
            output=f'{target_dir}/model.onnx', 
            opset=11,
            tokenizer=self.args.tokenizer, 
            pipeline_name='ner'
               )
        
        
        df = pd.DataFrame(eval_history)
        df.to_csv(f"{target_dir}/history.csv", index=False)
        
        # 마지막에 체크포인트 제거
        shutil.rmtree(self.args.output_dir)
        
        return df

def main(param):
    """search space of hyperparams
    max_seq_len : max sequence length of input
    """
    # set up basic arguments
    parser = HfArgumentParser((ModelArguments, SentimentExtractionDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # json config path
        print(parser.parse_json_file(json_file=os.path.abspath(sys.argv[1])))
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    #### get params from nni ########################################################
    if param != {}:
        # PLM names
        if param["model_type"]=="kobert":
            data_collator = hf_efficient_data_collator
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            
#         elif param["model_type"]=="distilkobert":
#             model_type = param["model_type"]
#             model_args.model_name_or_path = "monologg/distilkobert"
            
        elif param["model_type"]=="hanbert":
            data_collator = hf_efficient_data_collator
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            
        elif param["model_type"]=="koelectra-base":
            data_collator = hf_efficient_data_collator
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            
        elif param["model_type"]=="koelectra-small":
            data_collator = hf_efficient_data_collator
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            
        elif param["model_type"]=="koelectra-base-v2":
            data_collator = hf_efficient_data_collator
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            
        elif param["model_type"]=="koelectra-small-v2":
            data_collator = hf_efficient_data_collator
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            
        elif param["model_type"]=="kocharelectra-base":
            data_collator = hf_efficient_data_collator
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            
        elif param["model_type"]=="kcbert-base":
            data_collator = hf_efficient_data_collator
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            
        elif param["model_type"]=="kcbert-large":
            data_collator = hf_efficient_data_collator
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]

        elif param["model_type"]=="hanbert-crf":
            data_collator = None
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            data_args.pad_token_label_id = 0
            
        elif param["model_type"]=="koelectra-base-crf":
            data_collator = None
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            data_args.pad_token_label_id = 0
            
        elif param["model_type"]=="kocharelectra-base-crf":
            data_collator = None
            model_type = param["model_type"]
            model_args.model_name_or_path = MODEL_PATH_MAP[model_type]
            data_args.pad_token_label_id = 0
        
        # seq strategy
        data_args.max_seq_length = param["max_seq_length"]
        
        # 학습 strategy
        training_args.per_device_train_batch_size = param["per_device_train_batch_size"]
        training_args.learning_rate = param["learning_rate"]
#         training_args.weight_decay = param["weight_decay"]
        
        # Hedging the OOM
        if training_args.per_device_train_batch_size == 32:
            training_args.per_device_train_batch_size //= 2
            training_args.gradient_accumulation_steps *= 2
            
        if training_args.per_device_train_batch_size == 64:
            training_args.per_device_train_batch_size //= 4
            training_args.gradient_accumulation_steps *= 4
            
        if 'large' in param["model_type"]: #?
            training_args.per_device_train_batch_size //= 4
            training_args.gradient_accumulation_steps *= 4
        
        if data_args.max_seq_length > 256:
            training_args.per_device_train_batch_size //= 2
            training_args.gradient_accumulation_steps *= 2
        

        
    #################################################################################
    
    #### nni output_dir overwrite the 그냥 output_dir
    training_args.output_dir = f"{training_args.output_dir}/{nni.get_experiment_id()}-{nni.get_trial_id()}"
    ###################
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    training_args.seed = param['seed']
    set_seed(training_args.seed)
    
        
    #################################################################### refactoring
    from models.utils import label_map
    
    label_map = [label for i, label in enumerate(label_map) if (data_args.pad_token_label_id !=-100) or (i!=0)]
    # Load Pretrained model and Tokenizer
    config = CONFIG_CLASSES[model_type].from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(label_map),
        id2label={i:label for i, label in enumerate(label_map)},
        label2id={label:i for i, label in enumerate(label_map)}
    )
    tokenizer = TOKENIZER_CLASSES[model_type].from_pretrained(
        model_args.model_name_or_path,
    )
    model = MODEL_FOR_TOKEN_CLASSIFICATION[model_type].from_pretrained(
        model_args.model_name_or_path,
        config=config
    )
    
    
    #################################################################### refactoring
    # tokenizer 추가 for onnx export
    training_args.tokenizer = tokenizer
    
    # Get datasets
    if training_args.do_train:
        train_dataset = SentimentExtractionDataset(data_args, tokenizer=tokenizer, mode = Split.train)
    if training_args.do_eval:
        eval_dataset = SentimentExtractionDataset(data_args, tokenizer=tokenizer, mode = Split.dev)
    if training_args.do_predict:
        data_args.data_dir = "data/test"
        test_dataset = SentimentExtractionDataset(data_args, tokenizer=tokenizer, mode = Split.test)

    # reset logging, eval and save step as EPOCHS explicitly
    steps_per_epoch = int(np.ceil(len(train_dataset) / (training_args.train_batch_size * training_args.gradient_accumulation_steps))) 
    training_args.logging_steps = steps_per_epoch
    training_args.save_steps  = steps_per_epoch
    training_args.eval_steps = steps_per_epoch

    # tensorboard
    tb_writer = SummaryWriter(f'{training_args.output_dir}/tblogs')
    
    # init trainer
    trainer = NNITrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tb_writer = tb_writer
    )
    
    # Training
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path 
                      if os.path.isdir(model_args.model_name_or_path) else None)
    
    # final eval
    train_time = (datetime.today() + timedelta(hours=9)).strftime("%Y-%m-%d")
    backbone = model_args.model_name_or_path.split('/')[-1]
    hyperparams = ",".join([f"{k}={param[k]}" for k in param])
    app_key = f"{train_time}/{backbone}/{hyperparams}"
    
    df = trainer.get_best_model(metric='token-f1', app_key=app_key, app_name='sentiment-extractor') # hard-coded
    
    # nni
    nni.report_final_result(df['eval_token-f1'].max()) # best case
    

if __name__ == '__main__':
    try:
        param_trials = nni.get_next_parameter()
        logger.debug(param_trials)
        main(param_trials)
    except Exception as exception:
        logger.exception(exception)
        raise