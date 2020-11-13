import logging
import os
import json
from typing import List, Union

from finetune import InputExample, Split, TokenClassificationTask

logger = logging.getLogger(__name__)

class SentimentTagging(TokenClassificationTask):
    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        """"""
        examples = []
        fn = f"{data_dir}/{mode.value}.json"
        with open(fn) as f:
            docs=json.load(f)
            
        for i, doc in enumerate(docs):
            guid = f"{mode.value} {doc['_id']}"
            words = doc['_source']['text'].split()
            labels = doc['_source']['label_list'].split()
            
            assert len(words) == len(labels)
                            
            if i % 2000 == 0:
                logger.info(json.dumps(doc, ensure_ascii=False, indent=2))
            
            examples.append(InputExample(guid=guid, words=words, labels=labels))
        return examples
    
    def get_labels(self, path: str = None) -> List[str]:
        if path:
            with open(path) as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return ["O", "긍정-B", "긍정-I", "부정-B", "부정-I"]
