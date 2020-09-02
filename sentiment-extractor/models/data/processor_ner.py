import os
import json
import copy 
import logging

import re
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from transformers.data.processors import DataProcessor, InputExample
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

    
def ner_convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        task,
        pad_token_label_id=-100,
):
    ## crf layer 사용 유무에 따라서 labeling 방법이 달라짐.
    # if not use_crf -> ["O", "긍정-B", "긍정-I", "부정-B", "부정-I"] + ignore_index = -100 (for nn.CrossEntropy)
    # if use_crf -> ["PAD", "O", "긍정-B", "긍정-I", "부정-B", "부정-I"] + ignore_index = 0 (for use crf layer)
    
    if pad_token_label_id == -100 :
        label_list = ner_processors[task]().get_labels()[1:] # drop "PAD"
        
    elif pad_token_label_id == 0 :
        label_list = ner_processors[task]().get_labels()
        
    label_map = {label: i for i, label in enumerate(label_list)}
        
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example {} of {}".format(ex_index, len(examples)))

        tokens = []
        label_ids = []

        for word, label in zip(example.text_a, example.label):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # Add [SEP]
        tokens += [tokenizer.sep_token]
        label_ids += [pad_token_label_id]

        # Add [CLS]
        tokens = [tokenizer.cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

#         padding_length = max_seq_length - len(input_ids)
#         input_ids += [tokenizer.pad_token_id] * padding_length
#         attention_mask += [0] * padding_length
#         token_type_ids += [0] * padding_length
#         label_ids += [pad_token_label_id] * padding_length

#         assert len(input_ids) == max_seq_length
#         assert len(attention_mask) == max_seq_length
#         assert len(token_type_ids) == max_seq_length
#         assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids)
        )
    return features



class SentimentExtractionProcessor(DataProcessor):
    """Processor for the Sentiment Extraction"""
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """NER labels"""
        return ["PAD", "O", "긍정-B", "긍정-I", "부정-B", "부정-I"]
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    @classmethod
    def _read_file(cls, input_file):
        """Read tsv file, and return words and label as list"""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset[1:]):
            words = data[1].split()
            labels = data[2].split()
            guid = "%s-%s" % (set_type, i)

            assert len(words) == len(labels)

            if i % 10000 == 0:
                logger.info(data)
            examples.append(InputExample(guid=guid, text_a=words, label=labels))
        return examples

ner_processors = {
    "sentiment-extraction": SentimentExtractionProcessor
}

# ner_tasks_num_labels = {
#     "sentiment-extraction": 5
# }