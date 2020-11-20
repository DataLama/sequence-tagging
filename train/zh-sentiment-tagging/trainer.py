import json
import logging
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from importlib import import_module
from itertools import chain

# model
import torch.nn as nn
import transformers
import nni
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    # 옮겨질수잇음
    TrainerState,
    TrainerControl,
    DefaultFlowCallback
)
from transformers.trainer_utils import is_main_process
from transformers import TrainerCallback

from finetune import (
    DataTrainingArguments, 
    ModelArguments,
    TokenClassificationDataset,
    TokenClassificationTask,
    DataCollatorForTokenClassification, 
    Split
)
from utils import compute_metrics

logger = logging.getLogger(__name__)

# 잘되면 finetune으로
METRICS = []
class NNiCallback(TrainerCallback):
    def __init__(self, hp_metric = None, greater_is_better=False):
        self.hp_metric = f"eval_{hp_metric}"
        self.greater_is_better = greater_is_better
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs.get('logs')
        if self.hp_metric in logs.keys():
            metric = logs.get(self.hp_metric)
            METRICS.append(metric)
            nni.report_intermediate_result(metric)

            
            
#     def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         if self.greater_is_better:
#             nni.report_final_result(max(self.metrics))
#         else:
#             nni.report_final_result(min(self.metrics))

class MAIN:
    """Generalized version of main function for Notebook and Python."""
    def __init__(self):
        # argments
        try:
            __IPYTHON__
        except NameError:
            self.model_args, self.data_args, self.training_args = self._argparse_script()
        else:
            self.model_args, self.data_args, self.training_args = self._argparse_notebook()
                
        # logging and seed
        self._init_logging_and_seed()
        
    def _argparse_notebook(self):
        """argparse in notebook is just used for test the code.(Hard-coded)"""
        
        model_args = ModelArguments(
            model_name_or_path = 'hfl/chinese-electra-180g-small-discriminator',
            task_type = 'SentimentTagging',
        )
        
        data_args = DataTrainingArguments(
            data_dir = 'data/step-01', 
            max_seq_length=512,
        )
        
        training_args = TrainingArguments(
            output_dir='ckpt/test',
            overwrite_output_dir=True, 
            do_train=True,
            do_eval=True,
            do_predict=True,
            evaluation_strategy = "steps",
            evaluate_during_training=True,
            logging_first_step=True,
            num_train_epochs=3.0,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,
            metric_for_best_model='token-f1',
            greater_is_better = True
        )
        
        return model_args, data_args, training_args
    
    def _argparse_script(self):
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        return model_args, data_args, training_args
    
    def _init_logging_and_seed(self):
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if is_main_process(self.training_args.local_rank) else logging.WARN,
        )

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(self.training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
        logger.info("Training/evaluation parameters %s", self.training_args)

        # Set seed before initializing model.
        set_seed(self.training_args.seed)
        

    def align_predictions(self, predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.label_map[label_ids[i][j]])
                    preds_list[i].append(self.label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(self, p: EvalPrediction) -> Dict:
        preds_list, out_label_list =  self.align_predictions(p.predictions, p.label_ids)

        ## token-level comparison
        y_true = list(chain(*out_label_list))
        y_pred = list(chain(*preds_list))

        y_true = [tok.split('-')[0] for tok in y_true]
        y_pred = [tok.split('-')[0] for tok in y_pred]
        ##

        return {
            "token-f1": f1_score(y_true, y_pred, average='macro'),
            "token-acc": accuracy_score(y_true, y_pred)
        }
    
    def _custom_config(self, model_args, num_labels, id2label, label2id):
        if "hfl/chinese-roberta" in model_args.model_name_or_path:
            """You have to use Bertclass - https://github.com/ymcui/Chinese-BERT-wwm/blob/master/README_EN.md"""
            config = BertConfig.from_pretrained(
                    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                    cache_dir=model_args.cache_dir,
            )            
        else:
            """You can load with AutoClass.
            - voidful/albert_chinese_xxlarge
            - hfl/chinese-macbert-large
            - hfl/chinese-electra-180g-large-discriminator
            """
            config = AutoConfig.from_pretrained(
                    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                    cache_dir=model_args.cache_dir,
                    )
        return config
    
    def _custom_tokenizer(self, model_args):
        if ("hfl/chinese-roberta" in model_args.model_name_or_path) or ("voidful/albert_chinese" in model_args.model_name_or_path):
            """You have to use Bertclass - https://github.com/ymcui/Chinese-BERT-wwm/blob/master/README_EN.md"""
            tokenizer = BertTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast,
            )            
        else:
            """You can load with AutoClass.
            - hfl/chinese-macbert-large
            - hfl/chinese-electra-180g-large-discriminator
            """
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast,
                    )
            
        return tokenizer
    
    def _custom_model(self, model_args, config):
        if "hfl/chinese-roberta" in model_args.model_name_or_path:
            model = BertForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        else:
            """You can load with AutoClass.
            - voidful/albert_chinese_xxlarge
            - hfl/chinese-macbert-large
            - hfl/chinese-electra-180g-large-discriminator
            """
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        return model
    


    def main(self, hp_params):
        model_args = self.model_args
        data_args = self.data_args
        training_args = self.training_args
        
        # arguments manipulation
        if nni.get_experiment_id() != 'STANDALONE':
            training_args.output_dir = f"{training_args.output_dir}/{nni.get_experiment_id()}-{nni.get_trial_id()}"
        model_args.model_name_or_path = hp_params['backbone']
        training_args.learning_rate = hp_params['learning_rate']
        training_args.seed = hp_params['seed']
        if hp_params["max_seq_length"] > 384:
            training_args.per_device_train_batch_size = 2
            training_args.per_device_eval_batch_size = 2
            training_args.gradient_accumulation_steps = 16
            data_args.max_seq_length = hp_params["max_seq_length"] 
        else:
            data_args.max_seq_length = hp_params["max_seq_length"] 
        
        
        
        # get token classification task instance
        module = import_module("tasks")
        try:
            token_classification_task_clazz = getattr(module, model_args.task_type)
            token_classification_task: TokenClassificationTask = token_classification_task_clazz()
        except AttributeError:
            raise ValueError(
                f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
                f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
            )
        
        # label
        labels = token_classification_task.get_labels(data_args.labels)
        label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
        num_labels = len(labels)
        self.label_map = label_map
        
        # load pretrained model and tokenizer
        config = self._custom_config(model_args=model_args, 
                                     num_labels=num_labels, 
                                     id2label=label_map, 
                                     label2id={label: i for i, label in enumerate(labels)})
        tokenizer = self._custom_tokenizer(model_args=model_args)
        model = self._custom_model(model_args=model_args, config=config)
        
        
#         config = AutoConfig.from_pretrained(
#         model_args.config_name if model_args.config_name else model_args.model_name_or_path,
#         num_labels=num_labels,
#         id2label=label_map,
#         label2id={label: i for i, label in enumerate(labels)},
#         cache_dir=model_args.cache_dir,
#         )
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
#             cache_dir=model_args.cache_dir,
#             use_fast=model_args.use_fast,
#         )
#         model = AutoModelForTokenClassification.from_pretrained(
#             model_args.model_name_or_path,
#             from_tf=bool(".ckpt" in model_args.model_name_or_path),
#             config=config,
#             cache_dir=model_args.cache_dir,
#         )
        
        # get dataset and data_collator
        train_dataset = (
            TokenClassificationDataset(
                token_classification_task=token_classification_task,
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.train,
            )
            if training_args.do_train
            else None
        )
        eval_dataset = (
            TokenClassificationDataset(
                token_classification_task=token_classification_task,
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.dev,
            )
            if training_args.do_eval
            else None
        )
        data_collator = DataCollatorForTokenClassification(tokenizer)
        
        # callbacks
        callbacks = [
            NNiCallback(hp_metric=training_args.metric_for_best_model, greater_is_better=training_args.greater_is_better)
        ]
        
        # reset logging, eval and save step as EPOCHS explicitly
        steps_per_epoch = int(np.ceil(len(train_dataset) / (training_args.train_batch_size * training_args.gradient_accumulation_steps))) 
        training_args.logging_steps = steps_per_epoch
        training_args.save_steps  = steps_per_epoch
        training_args.eval_steps = steps_per_epoch

        
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            compute_metrics=self.compute_metrics,
        )
        
        # Training
        if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
            trainer.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if trainer.is_world_master():
                tokenizer.save_pretrained(training_args.output_dir)

        # Evaluation
        results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            result = trainer.evaluate()

            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key, value in result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

                results.update(result)

        # Predict
        if training_args.do_predict:
            test_dataset = TokenClassificationDataset(
                token_classification_task=token_classification_task,
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.test,
            )

            predictions, label_ids, metrics = trainer.predict(test_dataset)
            preds_list, _ = self.align_predictions(predictions, label_ids)

            output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
            if trainer.is_world_master():
                with open(output_test_results_file, "w") as writer:
                    for key, value in metrics.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            # Save predictions
            output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
            if trainer.is_world_master():
                with open(output_test_predictions_file, "w") as writer:
                    with open(os.path.join(data_args.data_dir, "test.json"), "r") as f:
                        docs = json.load(f)
                    for doc, preds in zip(docs, preds_list):
                        text = doc['_source']['text']
                        labels = doc['_source']['label_list']
                        preds = ' '.join(preds)
                        print(f"{text}\t{labels}\t{preds}", file=writer)
                        
                        
        # nni final result
        if training_args.greater_is_better:
            nni.report_final_result(max(METRICS))
        else:
            nni.report_final_result(min(METRICS))

                        
                        
if __name__=="__main__":
    try:
        param_trials = nni.get_next_parameter()
        logger.debug(param_trials)
        MAIN().main(param_trials)
    except Exception as exception:
        logger.exception(exception)
        raise
        
        
