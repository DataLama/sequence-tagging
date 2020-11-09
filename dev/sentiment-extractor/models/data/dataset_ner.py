import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, NewType, Any, Dict

from filelock import FileLock

from transformers import PreTrainedTokenizer, BatchEncoding
from .processor_ner import ner_convert_examples_to_features, ner_processors, InputFeatures

# torch
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence


InputDataClass = NewType("InputDataClass", Any)

logger = logging.getLogger(__name__)

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

@dataclass
class SentimentExtractionDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    
    If you use HfArgumentPasrser, we can turn this class 
    into argparse arguments to be albe to specify them on 
    the command line.
    """
    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(ner_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    
    pad_token_label_id: int = field(
        default=-100, metadata={"help": "nn.Crossentropy's default ignore_index is -100"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()
        
        
class SentimentExtractionDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        args: SentimentExtractionDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None
    ):
        self.args = args
        self.processor = ner_processors[args.task_name]()
        
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else self.args.data_dir,
             "cached_{}_{}_{}_ignore-{}".format(mode.value, tokenizer.__class__.__name__, str(self.args.max_seq_length), str(self.args.pad_token_label_id)),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not self.args.overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {self.args.data_dir}")
                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)                
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = ner_convert_examples_to_features(
                    examples,
                    tokenizer,
                    self.args.max_seq_length,
                    self.args.task_name,
                    pad_token_label_id=self.args.pad_token_label_id
                )
                # eval인 경우 추가하기.
                if mode != Split.train:
                    pad_features = []
                    for i, feature in enumerate(self.features):
                        padding_length = self.args.max_seq_length - len(feature.input_ids)
                        
                        input_ids = feature.input_ids
                        attention_mask = feature.attention_mask
                        token_type_ids = feature.token_type_ids
                        label_ids = feature.label_ids
                        
                        input_ids += [tokenizer.pad_token_id] * padding_length
                        attention_mask += [0] * padding_length
                        token_type_ids += [0] * padding_length
                        label_ids += [self.args.pad_token_label_id] * padding_length
                        
                        assert len(feature.input_ids) == self.args.max_seq_length
                        assert len(feature.attention_mask) == self.args.max_seq_length
                        assert len(feature.token_type_ids) == self.args.max_seq_length
                        assert len(feature.label_ids) == self.args.max_seq_length
                        
                        pad_features.append(
                            InputFeatures(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          label_ids=label_ids)
                        )
                    features = pad_features
                
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def hf_efficient_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """Efficient data collator for Huggingface style dataset.
    It is reference from transformers.data.data_collator.default_data_collator
    
    main_functions
    - from python list to torch.Tensor
    - pad_sequences for variable length input
    """
    
    # List[InputFeatures] -> List[Dict]
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]
  
    # Define base variable
    first = features[0] # first data를 proxy로 사용함.
    batch = {}
    
    # Special handling for labels.
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            padded_labels = pad_sequence([f["label_ids"] for f in features], batch_first=True, padding_value=-100)
            batch["labels"] = padded_labels
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            labels = [torch.tensor(f["label_ids"], dtype=dtype) for f in features]
            padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            batch["labels"] = padded_labels

    # Special handling for Inputs
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                padded_inputs = pad_sequence([f[k] for f in features], batch_first=True, padding_value=0)
                batch[k] = padded_inputs
            else:
                inputs = [torch.tensor(f[k], dtype=torch.long) for f in features]
                padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
                batch[k] = padded_inputs
                
    print(padded_inputs.shape)
    return batch